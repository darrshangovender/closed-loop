"""P4a: label resolution, 0001 section 6.2 exactly. SYNTHETIC data only.

The table below is hand-built, one row per rule and boundary. The mutation the packet asks for (swap
``>=`` and ``>`` in the censored rules) turns the rows at exactly ``t + lo`` and ``t + hi`` red.
"""

import dataclasses
import math

import pytest

from closed_loop.perception.labels import (
    Checkpoint,
    Label,
    LabelStatus,
    RulInterval,
    label_status,
    learner_view,
    resolve_labels,
)

H, M, LH, U = LabelStatus.HIT, LabelStatus.MISS, LabelStatus.LOWER_HIT, LabelStatus.UNKNOWN
INF = math.inf

# (lo, hi, checkpoint day, end day, censored, expected status, why)
CASES = [
    # interval [5, 20] at day 10: the window is days 15 to 30
    (5, 20, 10, 15.0, False, H, "failure on the lower edge is a hit"),
    (5, 20, 10, 30.0, False, H, "failure on the upper edge is a hit"),
    (5, 20, 10, 22.0, False, H, "failure inside"),
    (5, 20, 10, 14.99, False, M, "failure just before the window"),
    (5, 20, 10, 30.01, False, M, "failure just after the window"),
    (5, 20, 10, 10.0, False, M, "failure on the checkpoint day, lower bound is 5"),
    (5, 20, 10, 30.0, True, M, "censored exactly at t + hi: T > tau >= t + hi"),
    (5, 20, 10, 29.99, True, LH, "censored just before t + hi"),
    (5, 20, 10, 15.0, True, LH, "censored exactly at t + lo: the lower bound held"),
    (5, 20, 10, 14.99, True, U, "censored just before t + lo"),
    (5, 20, 10, 10.0, True, U, "replaced on the checkpoint day, lower bound is 5"),
    (5, 20, 10, 50.0, True, M, "censored long after the upper bound"),
    (5, 20, 10, 20.0, True, LH, "censored inside the window"),
    # interval reaching below zero: [-3, 8] at day 10
    (-3, 8, 10, 10.0, False, H, "failure today inside a negative-lower interval"),
    (-3, 8, 10, 10.0, True, LH, "replaced today: tau >= t + lo holds for a negative lo"),
    (-3, 8, 10, 18.0, True, M, "censored on t + hi with a negative lo"),
    (-3, 8, 10, 17.5, True, LH, "censored just inside with a negative lo"),
    # unbounded sides
    (5, INF, 10, 1e6, False, H, "infinite upper bound, far failure"),
    (5, INF, 10, 1e6, True, LH, "infinite upper bound, censored far on: never a miss"),
    (-INF, 8, 10, 10.0, True, LH, "infinite lower bound, censored today"),
    (-INF, INF, 10, 10.0, False, H, "the whole line always hits"),
    # zero width, lo == hi == 6 at day 10 (the window is the single day 16)
    (6, 6, 10, 16.0, False, H, "zero width, failure exactly on it"),
    (6, 6, 10, 16.0, True, M, "zero width, censored on it: T > 16, so the upper bound failed"),
    (6, 6, 10, 15.9, True, U, "zero width, censored just before"),
    (6, 6, 10, 16.1, False, M, "zero width, failure just after"),
]


@pytest.mark.parametrize(("lo", "hi", "day", "end", "censored", "want", "why"), CASES)
def test_status_table(lo, hi, day, end, censored, want, why):
    assert label_status(lo, hi, day, end, censored) is want, why


def test_the_table_has_at_least_twenty_cases_and_uses_every_status():
    assert len(CASES) >= 20
    assert {c[5] for c in CASES} == set(LabelStatus)


def cp(day, lo=5.0, hi=20.0, life=3, asset="PU-201", **kw):
    return Checkpoint(life, asset, day, lo, 12.0, hi, 4.0, 0.9, **kw)


class TestResolvedAt:
    """``end_day`` is the last time the component ran; ``resolved_at`` is when the learner knows."""

    def test_labels_resolve_when_the_learner_knows_and_keep_the_statuses_of_the_end_day(self):
        cps = [cp(10), cp(20, lo=2.0, hi=9.0)]
        a = resolve_labels(cps, end_day=24.0, censored=True, audit=False)
        b = resolve_labels(cps, end_day=24.0, censored=True, audit=False, resolved_at=27.3)
        assert [x.status for x in a] == [x.status for x in b] == [LH, LH]
        assert {x.resolved_day for x in a} == {24}
        assert {x.resolved_day for x in b} == {28}
        assert all(x.end_day == 24.0 for x in b)

    def test_resolved_at_cannot_precede_the_end(self):
        with pytest.raises(ValueError, match="resolved_at"):
            resolve_labels([cp(10)], 24.0, True, False, resolved_at=23.9)
        (lab,) = resolve_labels([cp(10)], 24.0, True, False, resolved_at=24.0)
        assert lab.resolved_day == 24


class TestResolve:
    def test_each_checkpoint_gets_the_status_of_its_own_interval(self):
        cps = [cp(10), cp(20, lo=2.0, hi=9.0), cp(25, lo=0.0, hi=3.0), cp(24, lo=5.0, hi=9.0)]
        # life ends in a replacement at day 30: t + hi is 30, 29, 28 and 33;
        # t + lo is 15, 22, 25 and 29
        labels = resolve_labels(cps, end_day=30.0, censored=True, audit=False)
        assert [lab.status for lab in labels] == [M, M, M, LH]
        # replaced at day 28: now t + hi is out of reach for the first two, 28 >= 28 for the third,
        # and the fourth has not reached its lower bound (29)
        labels = resolve_labels(cps, end_day=28.0, censored=True, audit=False)
        assert [lab.status for lab in labels] == [LH, LH, M, U]

    def test_a_failure_gives_exact_labels_and_no_censored_flag(self):
        labels = resolve_labels([cp(10), cp(28)], end_day=29.0, censored=False, audit=True)
        assert [lab.status for lab in labels] == [H, M]  # 19 in [5, 20]; 1 not in [5, 20]
        assert not any(lab.censored for lab in labels)
        assert [lab.observed_rul for lab in labels] == [19.0, 1.0]

    def test_a_censored_label_has_no_observed_rul(self):
        (lab,) = resolve_labels([cp(10)], end_day=18.0, censored=True, audit=False)
        assert lab.censored and lab.observed_rul is None

    def test_fields_are_carried_over_and_the_order_is_kept(self):
        cps = [cp(12, est_status="not_degrading"), cp(11), cp(13, est_status="prior")]
        labels = resolve_labels(cps, end_day=40.5, censored=True, audit=True)
        assert [lab.checkpoint_day for lab in labels] == [12, 11, 13]
        assert [lab.prior for lab in labels] == [True, False, True]
        assert [lab.blind for lab in labels] == [False, False, True]
        assert all(lab.audit and lab.censored for lab in labels)
        assert all(lab.life_id == 3 and lab.asset_id == "PU-201" for lab in labels)
        assert all(lab.end_day == 40.5 and lab.level == 0.9 for lab in labels)
        assert labels[0].lo == 5.0 and labels[0].hi == 20.0
        assert labels[0].m == 12.0 and labels[0].sigma == 4.0

    def test_resolved_day_is_the_first_whole_day_at_or_after_the_end(self):
        assert resolve_labels([cp(10)], 40.0, True, False)[0].resolved_day == 40
        assert resolve_labels([cp(10)], 40.01, True, False)[0].resolved_day == 41
        assert resolve_labels([cp(10)], 40.99, False, False)[0].resolved_day == 41

    def test_no_checkpoints_no_labels(self):
        assert resolve_labels([], 10.0, True, False) == []

    def test_checkpoints_of_two_lives_are_refused(self):
        with pytest.raises(ValueError, match="one life"):
            resolve_labels([cp(10), cp(11, life=4)], 30.0, True, False)
        with pytest.raises(ValueError, match="one life"):
            resolve_labels([cp(10), cp(11, asset="PU-202")], 30.0, True, False)

    def test_a_checkpoint_after_an_observed_failure_is_refused(self):
        with pytest.raises(ValueError, match="after the failure"):
            resolve_labels([cp(31)], 30.0, False, False)
        resolve_labels([cp(30)], 30.0, False, False)  # on the failure day is fine

    def test_a_checkpoint_after_the_component_stopped_is_unknown(self):
        """The daily loop keeps producing intervals for an asset stopped for its replacement. The
        rules give those checkpoints UNKNOWN (no ``lo >= 0`` can be reached), not an error."""
        labs = resolve_labels([cp(30), cp(31), cp(35, lo=0.0, hi=2.0)], 30.0, True, False)
        assert [x.status for x in labs] == [U, U, U]
        # a lower bound below zero is vacuous for a non-negative RUL: LOWER_HIT, as the rule says
        (vac,) = resolve_labels([cp(31, lo=-2.0, hi=9.0)], 30.0, True, False)
        assert vac.status is LH

    def test_nan_end_day_is_refused(self):
        with pytest.raises(ValueError, match="NaN"):
            resolve_labels([cp(10)], math.nan, True, False)
        with pytest.raises(ValueError, match="NaN"):
            label_status(5.0, 20.0, 10, math.nan, True)


class TestCensoringFollowsFromTheRules:
    """A replacement before the lower bound is uninformative because of the rules, not a special
    case: sweep the replacement day and watch the label move through the three censored statuses."""

    def test_sweeping_tau_gives_unknown_then_lower_hit_then_miss(self):
        statuses = [label_status(5.0, 20.0, 10, 10.0 + c, True) for c in range(0, 40)]
        first_lh = statuses.index(LH)
        first_miss = statuses.index(M)
        assert all(s is U for s in statuses[:first_lh])
        assert all(s is LH for s in statuses[first_lh:first_miss])
        assert all(s is M for s in statuses[first_miss:])
        assert (first_lh, first_miss) == (5, 20)

    def test_replacing_one_day_before_the_lower_bound_is_always_unknown(self):
        """The bound rule of maintenance-agent: replace at lower bound minus one day."""
        for lo in (3.0, 7.5, 12.0, 30.0):
            for day in (0, 17, 60):
                tau = day + lo - 1.0
                assert label_status(lo, lo + 15.0, day, tau, True) is U

    def test_no_censored_label_is_ever_a_hit(self):
        for c in range(0, 60):
            for lo, hi in ((0.0, 5.0), (5.0, 20.0), (-4.0, 3.0), (10.0, 10.0)):
                assert label_status(lo, hi, 0, float(c), True) is not H


class TestLearnerView:
    def test_the_view_drops_the_truth_and_nothing_else(self):
        (lab,) = resolve_labels([cp(10, true_rul=21.0, health_bin=2)], 31.0, False, False)
        assert lab.true_rul == 21.0
        view = learner_view(lab)
        assert view.true_rul is None
        assert dataclasses.replace(view, true_rul=21.0) == lab

    def test_a_label_without_truth_is_unchanged(self):
        (lab,) = resolve_labels([cp(10)], 31.0, False, False)
        assert lab.true_rul is None and learner_view(lab) == lab

    def test_labels_are_frozen(self):
        (lab,) = resolve_labels([cp(10)], 31.0, False, False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            lab.status = M  # type: ignore[misc]
        assert isinstance(lab, Label)


class TestValueObjects:
    def test_rul_interval_refuses_nonsense(self):
        RulInterval(-1.0, 3.0, 9.0, 0.9, False)
        RulInterval(-INF, 3.0, INF, 0.9, True)
        with pytest.raises(ValueError, match="empty"):
            RulInterval(5.0, 3.0, 4.0, 0.9, False)
        with pytest.raises(ValueError, match="NaN"):
            RulInterval(math.nan, 3.0, 4.0, 0.9, False)
        with pytest.raises(ValueError, match="level"):
            RulInterval(1.0, 3.0, 4.0, 1.0, False)

    def test_checkpoint_refuses_nonsense_and_flags_prior_and_blind(self):
        with pytest.raises(ValueError, match="sigma"):
            Checkpoint(0, "PU-201", 3, 1.0, 2.0, 3.0, 0.0, 0.9)
        with pytest.raises(ValueError, match="empty"):
            Checkpoint(0, "PU-201", 3, 4.0, 2.0, 3.0, 1.0, 0.9)
        a = cp(5)
        b = cp(5, est_status="prior")
        c = cp(5, est_status="insufficient")
        assert (a.prior, a.blind) == (False, False)
        assert (b.prior, b.blind) == (True, True)
        assert (c.prior, c.blind) == (True, False)
