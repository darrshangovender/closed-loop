"""``eval.claims``: a hand-worked win, loss and inconclusive case per claim (prereg section 7).

Each case states its numbers in the comment and the verdict follows by reading the registered rule,
not by running the code. The boundaries get their own cases: "at least" includes the number, "below"
and "above" do not.
"""

from __future__ import annotations

import dataclasses

import pytest

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval.claims import (
    ADAPT_VARIANTS,
    COMPLETED,
    INCONCLUSIVE,
    LOSS,
    NOT_RUN,
    WIN,
    Est,
    SearchPairs,
    ShieldInputs,
    Verdict,
    c_adapt,
    c_cause,
    c_diag,
    c_mon,
    c_perc,
    c_plan,
    c_search,
    c_shield,
    c_together,
    claim_name,
    decide_pair,
    est_of,
    fmt,
    verdict_records,
    verdicts_from_records,
    verdicts_from_stored,
)
from closed_loop.eval.counts import ALPHA_PERC, ALPHA_PRIMARY
from closed_loop.eval.report import FORBIDDEN


def e(value: float, lo: float, hi: float, n: int = 500, **kw: int) -> Est:
    return Est(value, lo, hi, n, **kw)


# ----------------------------------------------------------------------------- C-PERC


class TestPerc:
    """Gap_B against the band [-0.03, +0.03]; V1 decides a win, V1 and the primary a loss."""

    def test_win_when_the_interval_of_v1_is_above_the_band(self) -> None:
        # V1: [0.04, 0.08] has lower bound 0.04 > 0.03, so the interval excludes the band.
        v = c_perc("pooled", e(0.06, 0.04, 0.08), e(0.0, -0.02, 0.02), "V3")
        assert v.label == WIN and "above" in v.headline and v.alpha == ALPHA_PERC

    def test_win_when_the_interval_is_below_the_band_and_the_sign_is_reported(self) -> None:
        v = c_perc("trend", e(-0.06, -0.08, -0.04), e(0.0, -0.01, 0.01), "V3")
        assert v.label == WIN and "below" in v.headline

    def test_loss_when_v1_and_the_primary_variant_are_inside_the_band(self) -> None:
        # V1 [-0.02, 0.02] and V3 [-0.01, 0.025] both lie inside [-0.03, 0.03]
        v = c_perc("pooled", e(0.0, -0.02, 0.02), e(0.01, -0.01, 0.025), "V3")
        assert v.label == LOSS and "not supported" in v.headline

    def test_inconclusive_when_v1_straddles_the_edge_of_the_band(self) -> None:
        # [0.0, 0.05] reaches past 0.03 but its lower bound is not above it
        v = c_perc("pooled", e(0.02, 0.0, 0.05), e(0.0, -0.01, 0.01), "V3")
        assert v.label == INCONCLUSIVE

    def test_inconclusive_when_only_v1_is_inside(self) -> None:
        v = c_perc("pooled", e(0.0, -0.02, 0.02), e(0.06, 0.04, 0.08), "V3")
        assert v.label == INCONCLUSIVE  # not a loss: the primary variant is outside the band

    def test_the_edges(self) -> None:
        assert (
            c_perc("pooled", e(0.05, 0.03, 0.07), e(0, -0.01, 0.01), "V3").label != WIN
        )  # lo = 0.03
        both = c_perc("pooled", e(0, -0.03, 0.03), e(0, -0.03, 0.03), "V3")
        assert both.label == LOSS  # an interval equal to the band is inside it

    def test_the_two_verdicts_have_their_own_names_and_the_stricter_level(self) -> None:
        a = c_perc("pooled", e(0, -0.01, 0.01), e(0, -0.01, 0.01), "V3")
        b = c_perc("trend", e(0, -0.01, 0.01), e(0, -0.01, 0.01), "V3")
        assert (a.claim, b.claim) == ("C-PERC.pooled", "C-PERC.trend")
        assert a.alpha == pytest.approx(0.05 / 18)

    def test_a_win_on_one_stratum_and_a_loss_on_the_other_stay_separate(self) -> None:
        win = c_perc("pooled", e(0.06, 0.04, 0.08), e(0, -0.01, 0.01), "V3")
        loss = c_perc("trend", e(0, -0.01, 0.01), e(0, -0.01, 0.01), "V3")
        assert (win.label, loss.label) == (WIN, LOSS)


# ----------------------------------------------------------------------------- C-ADAPT


class TestAdapt:
    def test_win(self) -> None:
        # paired difference [0.03, 0.07] has lower bound 0.03 >= 0.02; shifted c_e^lo has lower
        # bound 0.93 >= 0.92
        v = c_adapt("V3", e(0.05, 0.03, 0.07), e(0.95, 0.93, 0.97), e(0.95, 0.93, 0.97))
        assert v.label == WIN and v.claim == "C-ADAPT" and v.alpha == pytest.approx(0.05 / 9)

    def test_loss_when_it_is_no_better_than_split(self) -> None:
        # upper bound 0.015 < 0.02
        v = c_adapt("V3", e(0.0, -0.02, 0.015), e(0.95, 0.93, 0.97), e(0.95, 0.93, 0.97))
        assert v.label == LOSS and "below 0.02" in v.headline

    def test_loss_when_the_nominal_bound_shows_coverage_below_the_tolerance(self) -> None:
        # the win conditions hold, nominal c_e^lo is at most 0.915 < 0.92: the loss rule is a veto
        v = c_adapt("V3", e(0.05, 0.03, 0.07), e(0.95, 0.93, 0.97), e(0.91, 0.89, 0.915))
        assert v.label == LOSS and any("veto" in n for n in v.notes)

    def test_a_point_estimate_below_the_line_is_not_a_loss_if_the_bound_reaches_it(self) -> None:
        # nominal 0.91 but the upper bound is 0.95: the interval does not show the miss, so the
        # loss rule does not fire and the win conditions (which do not read nominal) decide
        v = c_adapt("V3", e(0.05, 0.03, 0.07), e(0.95, 0.93, 0.97), e(0.91, 0.85, 0.95))
        assert v.label == WIN

    def test_inconclusive_when_the_interval_straddles_the_margin(self) -> None:
        v = c_adapt("V3", e(0.02, -0.01, 0.05), e(0.95, 0.93, 0.97), e(0.95, 0.93, 0.97))
        assert v.label == INCONCLUSIVE

    def test_a_shifted_point_estimate_above_the_line_is_not_a_win_if_the_bound_is_below_it(
        self,
    ) -> None:
        # shifted c_e^lo 0.94 but its lower bound is 0.90 < 0.92: the bound does not show it
        v = c_adapt("V3", e(0.05, 0.03, 0.07), e(0.94, 0.90, 0.97), e(0.95, 0.93, 0.97))
        assert v.label == INCONCLUSIVE

    def test_inconclusive_when_shifted_coverage_is_short_but_nominal_is_fine(self) -> None:
        v = c_adapt("V3", e(0.05, 0.03, 0.07), e(0.90, 0.85, 0.95), e(0.95, 0.93, 0.97))
        assert v.label == INCONCLUSIVE

    def test_the_edges(self) -> None:
        # a lower bound of exactly 0.92 and a margin of exactly 0.02 are wins ("at least")
        assert (
            c_adapt("V3", e(0.04, 0.02, 0.06), e(0.94, 0.92, 0.96), e(0.92, 0.9, 0.95)).label == WIN
        )
        # an upper bound of exactly 0.02 and of exactly 0.92 are not losses ("below")
        assert (
            c_adapt("V3", e(0.02, -0.01, 0.02), e(0.95, 0.93, 0.97), e(0.95, 0.93, 0.97)).label
            != LOSS
        )
        assert (
            c_adapt("V3", e(0.05, 0.03, 0.07), e(0.95, 0.93, 0.97), e(0.91, 0.9, 0.92)).label
            != LOSS
        )

    def test_a_secondary_variant_has_its_own_name(self) -> None:
        v = c_adapt(
            "V5", e(0.05, 0.03, 0.07), e(0.95, 0.93, 0.97), e(0.95, 0.93, 0.97), primary=False
        )
        assert v.claim == "C-ADAPT.V5"


# ----------------------------------------------------------------------------- C-CAUSE, C-DIAG


class TestCause:
    def test_win(self) -> None:
        # [-0.12, -0.05]: upper bound -0.05 < -1/30
        assert c_cause(e(-0.08, -0.12, -0.05, 300)).label == WIN

    def test_loss(self) -> None:
        assert c_cause(e(0.05, 0.01, 0.09, 300)).label == LOSS  # lower bound above 0

    def test_inconclusive(self) -> None:
        assert c_cause(e(-0.02, -0.06, 0.01, 300)).label == INCONCLUSIVE

    def test_one_day_is_not_enough_to_win_exactly_at_the_margin(self) -> None:
        assert c_cause(e(-0.05, -0.1, -1.0 / 30.0, 300)).label == INCONCLUSIVE
        assert c_cause(e(-0.05, -0.1, -0.034, 300)).label == WIN

    def test_it_is_labelled_favoured_by_construction(self) -> None:
        assert any("favoured by construction" in n for n in c_cause(e(0, -0.1, 0.1)).notes)


class TestDiag:
    def test_win(self) -> None:
        assert c_diag(e(0.10, 0.05, 0.15, 900)).label == WIN  # lower bound 0.05 > 0.02

    def test_loss(self) -> None:
        assert c_diag(e(-0.10, -0.15, -0.05, 900)).label == LOSS  # upper bound below 0

    def test_inconclusive(self) -> None:
        assert c_diag(e(0.01, -0.01, 0.03, 900)).label == INCONCLUSIVE

    def test_edges(self) -> None:
        assert c_diag(e(0.05, 0.02, 0.08)).label == INCONCLUSIVE  # lower bound must be above
        assert c_diag(e(-0.05, -0.08, 0.0)).label == INCONCLUSIVE  # upper bound must be below

    def test_the_unreversed_result_is_reported_and_decides_nothing(self) -> None:
        v = c_diag(e(0.10, 0.05, 0.15), e(-0.2, -0.25, -0.15))
        assert v.label == WIN and any(
            "unreversed" in n and "no decision rule" in n for n in v.notes
        )
        assert v.numbers["unreversed_diff.value"] == -0.2


# ----------------------------------------------------------------------------- C-PLAN


class TestPlan:
    def test_win(self) -> None:
        # cost difference [-0.08, -0.03]: upper bound -0.03 < -0.02; f_e difference [-0.05, 0.08]:
        # upper bound 0.08 is at most 0.1
        v = c_plan("P0", "B3-V3", e(-0.05, -0.08, -0.03), e(0.0, -0.05, 0.08), "nominal")
        assert v.label == WIN and v.claim == "C-PLAN.P0"

    def test_loss_when_the_planner_costs_more(self) -> None:
        assert c_plan("P0", "B3-V3", e(0.03, 0.01, 0.05), e(0, -0.1, 0.1), "nominal").label == LOSS

    def test_loss_when_more_episodes_fail_by_more_than_a_tenth(self) -> None:
        # the cost win would hold, but f_e differs by [0.15, 0.45]: lower bound 0.15 > 0.1
        v = c_plan("P0", "B3-V3", e(-0.05, -0.08, -0.03), e(0.3, 0.15, 0.45), "nominal")
        assert v.label == LOSS

    def test_inconclusive(self) -> None:
        v = c_plan("P0", "B3-V3", e(-0.01, -0.04, 0.02), e(0, -0.1, 0.1), "nominal")
        assert v.label == INCONCLUSIVE

    def test_edges(self) -> None:
        assert (
            c_plan("P0", "b", e(-0.03, -0.05, -0.02), e(0, -0.1, 0.1), "nominal").label
            == INCONCLUSIVE
        )
        assert c_plan("P0", "b", e(-0.03, -0.05, -0.021), e(0, -0.1, 0.1), "nominal").label == WIN
        # the safety leg is the upper bound: at most 0.1 wins, above it does not, and a lower
        # bound of exactly 0.1 is not yet a loss
        assert c_plan("P0", "b", e(-0.03, -0.05, -0.021), e(0.0, -0.1, 0.1), "nominal").label == WIN
        inc = c_plan("P0", "b", e(-0.03, -0.05, -0.021), e(0.1, 0.1, 0.2), "nominal")
        assert inc.label == INCONCLUSIVE
        assert c_plan("P0", "b", e(0.0, 0.0, 0.05), e(0, -0.1, 0.1), "nominal").label != LOSS

    def test_the_safety_leg_is_read_on_the_upper_bound(self) -> None:
        # the review: [0.09, 0.9] passed on its lower bound; it is nothing like safe
        v = c_plan("P0", "b", e(-0.05, -0.08, -0.03), e(0.4, 0.09, 0.9), "nominal")
        assert v.label == INCONCLUSIVE
        assert (
            c_plan("P0", "b", e(-0.05, -0.08, -0.03), e(0.4, 0.101, 0.9), "nominal").label == LOSS
        )

    def test_the_shifted_rows_are_secondary_and_named_so(self) -> None:
        v = c_plan("P1", "B3-S", e(-0.05, -0.08, -0.03), e(0, -0.05, 0.05), "shifted")
        assert v.claim == "C-PLAN.P1.shifted"

    def test_the_nominal_headline_prints_the_shifted_paired_numbers_beside_the_nominal_ones(
        self,
    ) -> None:
        """0003 A18: the paired mean of the planner changed sign under shift on 20 seeds; a
        reader of the nominal verdict alone must see that on the same line."""
        shifted = (e(0.07, 0.01, 0.13), e(0.2, 0.05, 0.35))
        v = c_plan(
            "P1",
            "B3-S",
            e(-0.05, -0.08, -0.03),
            e(0.0, -0.05, 0.08),
            "nominal",
            shifted,
        )
        assert fmt(-0.05) in v.headline and fmt(-0.08) in v.headline
        assert "under shift" in v.headline
        assert v.headline.index("under shift") > v.headline.index("nominal")
        assert fmt(0.07) in v.headline and fmt(0.01) in v.headline and fmt(0.13) in v.headline
        assert fmt(0.2) in v.headline and fmt(0.35) in v.headline
        assert v.numbers["shifted.cost_diff.hi"] == 0.13
        assert v.numbers["shifted.failure_diff.lo"] == 0.05

    def test_the_shifted_numbers_decide_nothing(self) -> None:
        cost, fail = e(-0.05, -0.08, -0.03), e(0.0, -0.05, 0.08)
        bad_shift = (e(0.5, 0.4, 0.6), e(0.9, 0.8, 1.0))
        with_shift = c_plan("P1", "B3-S", cost, fail, "nominal", bad_shift)
        without = c_plan("P1", "B3-S", cost, fail, "nominal")
        assert with_shift.label == without.label == WIN
        assert "under shift" not in without.headline
        assert not [k for k in without.numbers if k.startswith("shifted.")]

    def test_the_safety_leg_is_the_share_of_episodes_with_a_failure(self) -> None:
        # 0003 A16: f_e, not the count u_e. The rule text, the headline and the numbers say so
        v = c_plan("P0", "B3-V3", e(-0.05, -0.08, -0.03), e(0.02, -0.05, 0.08), "nominal")
        assert "f_e" in v.rule and "share of episodes with an unplanned failure" in v.headline
        assert v.numbers["failure_diff.hi"] == 0.08 and "unplanned_diff.hi" not in v.numbers
        assert "0.1" in v.rule and "-0.02" in v.rule  # the margins did not move


# ----------------------------------------------------------------------------- C-MON


class TestMon:
    def test_win(self) -> None:
        # no false alarm in 500 (upper bound 0.015 <= 0.02), detected in about 70% (lower 0.55)
        v = c_mon("proxy", e(0.0, 0.0, 0.015, k=0), e(0.7, 0.55, 0.85, 150, k=105))
        assert v.label == WIN and v.claim == "C-MON"

    def test_loss_on_false_alarms(self) -> None:
        assert c_mon("pess", e(0.2, 0.15, 0.3, k=100), e(0.9, 0.8, 0.95)).label == LOSS

    def test_loss_when_there_is_no_useful_power(self) -> None:
        # detection lower bound 0.05 < 0.1
        assert c_mon("opt", e(0.0, 0.0, 0.01, k=0), e(0.0, 0.0, 0.05, 150, k=0)).label == LOSS

    def test_inconclusive(self) -> None:
        # false alarm bound 0.01 fine; detection lower bound 0.3 is between 0.1 and 0.5
        assert c_mon("inf", e(0.0, 0.0, 0.01, k=0), e(0.5, 0.3, 0.7)).label == INCONCLUSIVE

    def test_edges(self) -> None:
        assert c_mon("m", e(0, 0, 0.02), e(0.6, 0.5, 0.7)).label == WIN  # both at the limit
        # a loss needs the lower bound of the false-alarm rate above 0.02, not its upper bound
        assert c_mon("m", e(0.03, 0.0201, 0.06), e(0.6, 0.5, 0.7)).label == LOSS
        assert c_mon("m", e(0.02, 0.02, 0.05), e(0.6, 0.5, 0.7)).label == INCONCLUSIVE
        assert c_mon("m", e(0, 0, 0.0201), e(0.6, 0.5, 0.7)).label == INCONCLUSIVE
        assert c_mon("m", e(0, 0, 0.01), e(0.2, 0.1, 0.3)).label == INCONCLUSIVE  # 0.1 is not below
        assert (
            c_mon("m", e(0, 0, 0.01), e(0.05, 0.0, 0.0999)).label == LOSS
        )  # upper bound below 0.1
        assert c_mon("m", e(0, 0, 0.01), e(0.05, 0.0, 0.1)).label == INCONCLUSIVE

    def test_a_perfect_monitor_is_not_a_loss_whatever_the_size_of_the_sample(self) -> None:
        # the review: 0 false alarms in 500 and 500 of 500 detections returned a loss, because the
        # zero-event upper bound at this level is 0.024. It is inconclusive at 500 unshifted
        # episodes and a win at the registered 700 (upper bound 0.017) with 300 shifted ones.
        at_500 = c_mon("proxy", e(0.0, 0.0, 0.024, 500, k=0), e(1.0, 0.99, 1.0, 500, k=500))
        assert at_500.label == INCONCLUSIVE
        at_700 = c_mon("proxy", e(0.0, 0.0, 0.017, 700, k=0), e(1.0, 0.98, 1.0, 300, k=300))
        assert at_700.label == WIN

    def test_a_mode_that_was_not_chosen_has_its_own_name(self) -> None:
        assert c_mon("inf", e(0, 0, 0.01), e(0.6, 0.5, 0.7), chosen=False).claim == "C-MON.inf"


# ----------------------------------------------------------------------------- C-SEARCH


def search_inputs(cma_found: int = 9, rand_found: int = 3) -> tuple[Est, Est, Est]:
    """The descriptive inputs of a C-SEARCH verdict: the two find rates and their difference."""
    return (
        e(cma_found / 10, 0.0, 1.0, 10, k=cma_found),
        e(rand_found / 10, 0.0, 1.0, 10, k=rand_found),
        e((cma_found - rand_found) / 10, -1.0, 1.0, 10),
    )


class TestDecidePair:
    """A pair goes to the algorithm that found a violation when the other did not, or that needed
    fewer evaluations when both did; the rest are dropped (prereg section 7, 0003 A16)."""

    @pytest.mark.parametrize(
        ("cma", "rand", "want"),
        [
            (5, None, (True, False)),  # only CMA-ES found
            (None, 5, (False, True)),  # only random search found
            (30, 80, (True, False)),  # both, CMA-ES fewer
            (80, 30, (False, True)),  # both, random fewer
            (40, 40, (False, False)),  # both, the same number: dropped
            (None, None, (False, False)),  # neither: dropped
        ],
    )
    def test_the_rule(self, cma: int | None, rand: int | None, want: tuple[bool, bool]) -> None:
        assert decide_pair(cma, rand, 200) == want

    def test_a_find_beyond_the_budget_is_no_find(self) -> None:
        assert decide_pair(201, None, 200) == (False, False)
        assert decide_pair(201, 5, 200) == (False, True)
        assert decide_pair(5, 201, 200) == (True, False)
        assert decide_pair(200, None, 200) == (True, False)  # the last evaluation still counts

    def test_the_pair_counts_are_checked(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            SearchPairs(-1, 3)
        sp = SearchPairs(6, 2, 1, 1)
        assert (sp.decided, sp.total) == (8, 10)


class TestSearch:
    def test_win_on_a_clean_sweep_of_ten_pairs(self) -> None:
        v = c_search(SearchPairs(10, 0), *search_inputs(10, 0), None, 0)
        assert v.label == WIN and v.claim == "C-SEARCH" and v.alpha == ALPHA_PRIMARY
        assert v.numbers["share.value"] == 1.0 and v.numbers["pairs.cma"] == 10
        assert v.numbers["share.lo"] == pytest.approx(0.5551, abs=5e-4)  # (alpha / 2) ** (1 / 10)

    def test_loss_on_a_clean_sweep_the_other_way(self) -> None:
        v = c_search(SearchPairs(0, 10), *search_inputs(0, 10), None, 0)
        assert v.label == LOSS
        assert v.numbers["share.hi"] == pytest.approx(0.4449, abs=5e-4)

    def test_with_ten_decided_pairs_only_a_clean_sweep_decides(self) -> None:
        # 0003 A16: nine of ten is inconclusive, on either side
        assert c_search(SearchPairs(9, 1), *search_inputs(), None, 0).label == INCONCLUSIVE
        assert c_search(SearchPairs(1, 9), *search_inputs(), None, 0).label == INCONCLUSIVE
        assert c_search(SearchPairs(5, 5), *search_inputs(), None, 0).label == INCONCLUSIVE

    def test_a_clean_sweep_needs_nine_decided_pairs(self) -> None:
        # (alpha / 2) ** (1 / n) is 0.480 at eight and 0.520 at nine
        assert c_search(SearchPairs(8, 0, 2), *search_inputs(8, 0), None, 0).label == INCONCLUSIVE
        assert c_search(SearchPairs(9, 0, 1), *search_inputs(9, 0), None, 0).label == WIN
        assert c_search(SearchPairs(0, 9, 1), *search_inputs(0, 9), None, 0).label == LOSS
        assert c_search(SearchPairs(0, 8, 2), *search_inputs(0, 8), None, 0).label == INCONCLUSIVE

    def test_no_decided_pair_is_inconclusive(self) -> None:
        v = c_search(SearchPairs(0, 0, 7, 3), *search_inputs(0, 0), None, 0)
        assert v.label == INCONCLUSIVE and "no pair is decided" in v.headline
        assert c_search(SearchPairs(0, 0), *search_inputs(0, 0), None, 0).label == INCONCLUSIVE

    def test_the_dropped_pairs_are_counted_and_said(self) -> None:
        v = c_search(SearchPairs(10, 0, 4, 2), *search_inputs(10, 0), None, 0)
        assert v.label == WIN  # dropped pairs do not weaken a sweep of the decided ones
        assert (v.numbers["pairs.neither"], v.numbers["pairs.tied"]) == (4, 2)
        assert "16 run pairs" in v.headline and "neither found a violation: 4" in v.headline
        assert "same number of evaluations: 2" in v.headline

    def test_it_is_judged_at_the_claim_level_not_at_five_percent(self) -> None:
        # nine decided pairs, all CMA-ES: at 0.05 the lower bound is 0.66 and at 0.05 / 9 it is
        # 0.52; the interval is the claim level's, which is what the numbers carry
        v = c_search(SearchPairs(9, 0), *search_inputs(9, 0), None, 0)
        assert v.numbers["share.lo"] == pytest.approx((0.05 / 9 / 2) ** (1 / 9), abs=1e-9)
        assert v.numbers["share.lo"] < 0.6

    def test_the_descriptive_numbers_are_in_the_verdict_and_decide_nothing(self) -> None:
        # a find-rate difference whose interval excludes 0 does not win on its own
        diff = e(0.6, 0.1, 0.9, 10)
        inconclusive = c_search(
            SearchPairs(9, 1), e(0.9, 0.5, 1.0, 10, k=9), e(0.3, 0.1, 0.6, 10, k=3), diff, None, 0
        )
        assert inconclusive.label == INCONCLUSIVE
        assert "find-rate difference" in inconclusive.headline
        assert inconclusive.numbers["find_diff.lo"] == 0.1
        # and an evaluations difference is carried, not consulted
        evals = e(-0.3, -0.5, -0.1, 10)
        v = c_search(SearchPairs(5, 5), *search_inputs(10, 10), evals, 0)
        assert v.label == INCONCLUSIVE and v.numbers["evals_diff.hi"] == -0.1
        assert "evaluations to first violation" in v.headline

    def test_random_search_being_faster_is_no_longer_a_stated_exception(self) -> None:
        # the old note said an interval that excluded 0 for random search "is published as
        # inconclusive"; under A16 faster random search takes the pairs and can lose the claim
        faster = c_search(SearchPairs(0, 10), *search_inputs(10, 10), e(0.3, 0.1, 0.5, 10), 0)
        assert faster.label == LOSS
        assert not any("random search was faster" in n for n in faster.notes)

    def test_a_defect_in_r_in_is_reported_for_the_shield(self) -> None:
        v = c_search(SearchPairs(10, 0), *search_inputs(10, 0), None, 2)
        assert any("R_in: runs with a defect: 2" in n and "assumption_held" in n for n in v.notes)

    def test_no_defect_in_r_in_is_said_plainly(self) -> None:
        v = c_search(SearchPairs(10, 0), *search_inputs(10, 0), None, 0)
        assert [n for n in v.notes if n.startswith("R_in")] == ["R_in: runs with a defect: 0"]

    def test_the_rule_text_is_the_registered_one(self) -> None:
        v = c_search(SearchPairs(10, 0), *search_inputs(10, 0), None, 0)
        assert "Clopper-Pearson" in v.rule and "above 0.5" in v.rule and "below 0.5" in v.rule


# ----------------------------------------------------------------------------- perfect and tiny


class TestEveryClaimOnAPerfectInputAndOnTooLittleData:
    """0003 A15: every verdict function returns a win (completed for C-TOGETHER) on a perfect input,
    and inconclusive, never a loss, on a sample too small to decide."""

    WIDE = (0.0, -1.0, 1.0)  # an interval that decides nothing

    def wide(self, n: int = 2, **kw: int) -> Est:
        return e(*self.WIDE, n=n, **kw)

    def test_c_perc(self) -> None:
        assert c_perc("pooled", e(0.1, 0.08, 0.12), e(0.1, 0.08, 0.12), "V3").label == WIN
        assert c_perc("pooled", self.wide(), self.wide(), "V3").label == INCONCLUSIVE

    def test_c_adapt(self) -> None:
        perfect = c_adapt("V3", e(0.1, 0.08, 0.12), e(0.97, 0.95, 0.99), e(0.97, 0.95, 0.99))
        assert perfect.label == WIN
        tiny = c_adapt("V3", self.wide(), e(0.5, 0.0, 1.0, 2), e(0.5, 0.0, 1.0, 2))
        assert tiny.label == INCONCLUSIVE

    def test_c_cause(self) -> None:
        assert c_cause(e(-0.3, -0.35, -0.25)).label == WIN
        assert c_cause(self.wide()).label == INCONCLUSIVE

    def test_c_diag(self) -> None:
        assert c_diag(e(0.3, 0.25, 0.35)).label == WIN
        assert c_diag(self.wide()).label == INCONCLUSIVE

    def test_c_plan(self) -> None:
        perfect = c_plan("P1", "B3-S", e(-0.2, -0.25, -0.15), e(-0.5, -0.6, -0.4), "nominal")
        assert perfect.label == WIN
        tiny = c_plan("P1", "B3-S", self.wide(), self.wide(), "nominal")
        assert tiny.label == INCONCLUSIVE

    def test_c_mon(self) -> None:
        perfect = c_mon("proxy", e(0.0, 0.0, 0.017, 700, k=0), e(1.0, 0.98, 1.0, 300, k=300))
        assert perfect.label == WIN
        tiny = c_mon("proxy", e(0.0, 0.0, 1.0, 2, k=0), e(0.5, 0.0, 1.0, 2, k=1))
        assert tiny.label == INCONCLUSIVE

    def test_c_search(self) -> None:
        perfect = c_search(
            SearchPairs(10, 0),
            e(1.0, 0.8, 1.0, 10, k=10), e(0.0, 0.0, 0.2, 10, k=0), e(1.0, 0.8, 1.0, 10), None, 0,
        )  # fmt: skip
        assert perfect.label == WIN
        tiny = c_search(
            SearchPairs(1, 1), e(0.5, 0.0, 1.0, 2, k=1), e(0.5, 0.0, 1.0, 2, k=1), self.wide(),
            None, 0,
        )  # fmt: skip
        assert tiny.label == INCONCLUSIVE

    def test_c_together(self) -> None:
        assert c_together({"F2-B3": (60, 60)}, (5, 10)).label == COMPLETED
        assert c_together({}, None).label == INCONCLUSIVE

    def test_c_shield(self) -> None:
        assert c_shield(shield_inputs()).label == WIN
        tiny = shield_inputs(
            shielded={"H-C1": e(0.0, 0.0, 1.0, 4, k=0), "H-G1": e(0.0, 0.0, 1.0, 4, k=0)},
            unshielded={"H-C0": e(0.5, 0.0, 1.0, 4, k=2), "H-G0": e(0.5, 0.0, 1.0, 4, k=2)},
        )
        assert c_shield(tiny).label == INCONCLUSIVE


# ----------------------------------------------------------------------------- C-TOGETHER


class TestTogether:
    def test_completed_is_the_label_and_never_win(self) -> None:
        v = c_together({"F2-B3": (60, 60), "F2-P0": (60, 60), "F2-P1": (60, 60)}, (3, 10))
        assert v.label == COMPLETED == "completed" and v.label != WIN and "3 of 10" in v.headline

    def test_a_low_confirmed_fraction_still_completes(self) -> None:
        # there is no numerical threshold: a low fraction is a finding about F1
        assert c_together({"F2-B3": (60, 60)}, (0, 20)).label == COMPLETED

    def test_loss_when_an_arm_cannot_complete_an_episode(self) -> None:
        v = c_together(
            {"F2-B3": (60, 60), "F2-P1": (0, 60)}, (3, 10), {"F2-P1": "orchestrator blocked"}
        )
        assert v.label == LOSS and "F2-P1" in v.headline and "orchestrator blocked" in v.headline

    def test_loss_when_an_arm_completes_only_some(self) -> None:
        assert c_together({"F2-B3": (58, 60)}, (3, 10)).label == LOSS

    def test_no_find_to_replay_is_said_as_zero_of_zero_and_still_completes(self) -> None:
        for confirmed in (None, (0, 0)):
            v = c_together({"F2-B3": (60, 60)}, confirmed)
            assert v.label == COMPLETED and "0 of 0" in v.headline
        assert c_together({}, None).label == INCONCLUSIVE  # nothing was tried at all


# ----------------------------------------------------------------------------- C-SHIELD


def shield_inputs(**kw: object) -> ShieldInputs:
    base: dict[str, object] = {
        "n_max": 500,
        "shielded": {"H-C1": e(0.0, 0.0, 0.009, k=0), "H-G1": e(0.0, 0.0, 0.009, k=0)},
        "unshielded": {"H-C0": e(0.4, 0.3, 0.5, k=200), "H-G0": e(0.3, 0.2, 0.4, k=150)},
        "defects": {"H-C1": 0, "H-G1": 0, "P1": 0},
        "ratios": {"H-C": 0.9, "H-G": 0.8, "P": 0.95},
        "search_defects": 0,
    }
    base.update(kw)
    return ShieldInputs(**base)  # type: ignore[arg-type]


class TestShield:
    def test_win(self) -> None:
        # 0 of 500 in both shielded arms, lower bounds 0.3 and 0.2 > 0.05 unshielded, no defect
        v = c_shield(shield_inputs())
        assert v.label == WIN and v.qualifier == "" and v.text == "win"
        assert "0 of 500" in v.headline and "do not show that it is zero" in v.headline

    def test_a_win_on_a_stopped_plant_is_labelled_vacuous(self) -> None:
        # P1 against P0 on dev: 0.093 / 0.990 = 0.094 < 0.5 (0003 A13)
        v = c_shield(shield_inputs(ratios={"H-C": 0.9, "H-G": 0.8, "P": 0.093 / 0.990}))
        assert v.label == WIN and v.text == "win, vacuous: the plant is mostly stopped"
        assert any("below 0.5 for P" in n for n in v.notes)

    def test_a_ratio_of_exactly_half_is_not_vacuous(self) -> None:
        v = c_shield(shield_inputs(ratios={"H-C": 0.5, "H-G": 0.5, "P": 0.5}))
        assert v.qualifier == ""

    def test_the_label_labels_and_decides_nothing_else(self) -> None:
        plain = c_shield(shield_inputs())
        stopped = c_shield(shield_inputs(ratios={"H-C": 0.1, "H-G": 0.1, "P": 0.1}))
        assert (
            plain.label == stopped.label == WIN
            and plain.numbers["n_max"] == stopped.numbers["n_max"]
        )

    def test_the_ratios_are_printed_whatever_the_verdict(self) -> None:
        v = c_shield(shield_inputs(defects={"H-C1": 1}))
        assert v.label == LOSS and any("ratio" in n for n in v.notes)

    def test_the_h_g_note_stands_beside_the_ratios_whatever_their_size(self) -> None:
        """0003 A18: the unshielded H-G arm leaves the crusher in WORK with no work order, so a
        ratio above 1 for H-G does not say the shield helped; the verdict says so beside it."""
        for ratio in (0.3, 0.9, 1.6):
            v = c_shield(shield_inputs(ratios={"H-C": 0.9, "H-G": ratio, "P": 0.9}))
            notes = [n for n in v.notes if n.startswith("H-G ratio:")]
            assert len(notes) == 1
            assert "no legal way out" in notes[0] and "does not say the shield helped" in notes[0]
            assert "not a stall of a safety layer" in notes[0]
            ratio_note = next(n for n in v.notes if "as a ratio of the same proposer" in n)
            assert v.notes.index(notes[0]) == v.notes.index(ratio_note) + 1

    def test_no_h_g_note_without_an_h_g_ratio(self) -> None:
        v = c_shield(shield_inputs(ratios={"H-C": 0.9, "P": 0.9}))
        assert not [n for n in v.notes if n.startswith("H-G ratio:")]

    def test_a_plant_only_event_is_not_called_a_contract_failure(self) -> None:
        """0003 A18, F4: a plant-only event has no spec event to label, so it is neither a defect
        nor a contract failure; the verdict splits the two."""
        shielded = {"H-C1": e(0.04, 0.0, 0.2, k=2), "H-G1": e(0.0, 0.0, 0.009, k=0)}
        v = c_shield(shield_inputs(shielded=shielded, plant_only={"H-C1": 2, "H-G1": 0}))
        assert v.label == INCONCLUSIVE
        assert "2 plant-only events" in v.headline
        assert (
            "nothing to attribute" in v.headline
            and "contract failures, not defects" not in v.headline
        )
        assert v.numbers["plant_only.H-C1"] == 2

    def test_a_mixed_arm_gets_both_readings(self) -> None:
        shielded = {"H-C1": e(0.06, 0.0, 0.2, k=3), "H-G1": e(0.0, 0.0, 0.009, k=0)}
        v = c_shield(shield_inputs(shielded=shielded, plant_only={"H-C1": 1}))
        assert (
            "2 attributed to a broken assumption of the contract (contract failures, not defects)"
            in v.headline
        )
        assert "1 plant-only events" in v.headline

    def test_without_the_plant_only_records_the_old_wording_stands(self) -> None:
        shielded = {"H-C1": e(0.04, 0.0, 0.2, k=2), "H-G1": e(0.0, 0.0, 0.009, k=0)}
        v = c_shield(shield_inputs(shielded=shielded))
        assert "these are contract failures, not defects" in v.headline

    def test_the_plant_only_count_is_read_from_the_stored_records(self) -> None:
        records = [PARAMS]
        for arm, k in (("H-C1", 2), ("H-G1", 0)):
            records.append(
                rec(claim_name("C-SHIELD", "viol_p13", arm), e(k / 500, 0.0, 0.2, 500, k=k))
            )
            records.append(
                rec(claim_name("C-SHIELD", "plant_only", arm), e(k / 500, 0.0, 0.2, 500, k=k))
            )
            records.append(
                rec(claim_name("C-SHIELD", "defects", arm), e(0.0, 0.0, 0.009, 500, k=0))
            )
        for arm in ("H-C0", "H-G0"):
            records.append(
                rec(claim_name("C-SHIELD", "viol_p13", arm), e(0.4, 0.3, 0.5, 500, k=200))
            )
        for pair in ("H-C", "H-G", "P"):
            records.append(rec(claim_name("C-SHIELD", "throughput_ratio", pair), e(0.9, 0.9, 0.9)))
        v = {x.claim: x for x in verdicts_from_records(records)}["C-SHIELD"]
        assert v.numbers.get("plant_only.H-C1") == 2
        assert "plant-only events" in v.headline

    def test_loss_on_one_defect(self) -> None:
        v = c_shield(shield_inputs(defects={"H-C1": 0, "H-G1": 0, "P1": 1}))
        assert v.label == LOSS and "P1" in v.headline and "unproven" in v.headline

    def test_loss_on_a_defect_found_by_the_r_in_search(self) -> None:
        assert c_shield(shield_inputs(search_defects=1)).label == LOSS

    def test_a_loss_has_no_vacuity_label(self) -> None:
        v = c_shield(shield_inputs(defects={"H-C1": 2}, ratios={"H-C": 0.1, "H-G": 0.1, "P": 0.1}))
        assert v.label == LOSS and v.qualifier == ""

    def test_inconclusive_when_a_violation_is_a_contract_failure_and_not_a_defect(self) -> None:
        # 3 of 500 violate P1-P3 in H-C1, all attributed to a broken assumption: no defect, no win
        sh = {"H-C1": e(0.006, 0.0, 0.03, k=3), "H-G1": e(0.0, 0.0, 0.009, k=0)}
        v = c_shield(shield_inputs(shielded=sh))
        assert v.label == INCONCLUSIVE and "contract failures, not defects" in v.headline

    def test_inconclusive_when_the_unshielded_arm_does_not_clear_five_percent(self) -> None:
        # lower bound 0.04 <= 0.05: the comparison has no teeth
        un = {"H-C0": e(0.1, 0.04, 0.2, k=50), "H-G0": e(0.3, 0.2, 0.4, k=150)}
        v = c_shield(shield_inputs(unshielded=un))
        assert v.label == INCONCLUSIVE and "no teeth" in v.headline

    def test_inconclusive_when_the_cell_stopped_before_n_max(self) -> None:
        sh = {"H-C1": e(0.0, 0.0, 0.05, n=200, k=0), "H-G1": e(0.0, 0.0, 0.009, k=0)}
        assert c_shield(shield_inputs(shielded=sh)).label == INCONCLUSIVE

    def test_inconclusive_when_an_episode_failed(self) -> None:
        sh = {"H-C1": e(0.0, 0.0, 0.009, k=0, failed=1), "H-G1": e(0.0, 0.0, 0.009, k=0)}
        assert c_shield(shield_inputs(shielded=sh)).label == INCONCLUSIVE

    def test_the_lower_bound_must_be_above_five_percent_not_at_it(self) -> None:
        un = {"H-C0": e(0.2, 0.05, 0.3, k=100), "H-G0": e(0.3, 0.2, 0.4, k=150)}
        assert c_shield(shield_inputs(unshielded=un)).label == INCONCLUSIVE

    def test_the_interlock_comparison_is_reported_and_can_be_a_tie(self) -> None:
        tie = {
            "H-C": (e(0.0, -0.02, 0.03), 0.9, 0.85, 12.0, 30.0),
            "H-G": (e(0.0, -0.02, 0.03), 0.8, 0.8, 9.0, 11.0),
        }
        v = c_shield(shield_inputs(interlock=tie))
        text = " ".join(v.notes)
        assert "the hand-written interlock matches the shield" in text
        assert (
            "interlock against shield, H-C" in text and "throughput 0.8500 against 0.9000" in text
        )
        assert v.numbers["throughput_interlock.H-C"] == 0.85

    def test_the_interlock_can_be_separated_or_undecided(self) -> None:
        rows = {
            "H-C": (e(0.3, 0.2, 0.4), 0.9, 0.5, 12.0, 30.0),  # lower bound 0.2 >= 0.05
            "H-G": (e(0.04, -0.2, 0.2), 0.8, 0.8, 9.0, 11.0),
        }
        text = " ".join(c_shield(shield_inputs(interlock=rows)).notes)
        assert "is separated from the shield" in text and "cannot be separated" in text
        assert "headline: the hand-written interlock matches" not in text

    def interlock_text(self, diff: Est) -> str:
        rows = {"H-C": (diff, 0.9, 0.9, 12.0, 12.0)}
        return " ".join(c_shield(shield_inputs(interlock=rows)).notes)

    def test_matches_needs_both_bounds_inside_five_percent(self) -> None:
        # interlock minus shield, in M-VIOL (P1-P3): both bounds within +-0.05 is a match
        assert "the interlock matches the shield" in self.interlock_text(e(0.0, -0.049, 0.049))
        # a bound beyond -0.05 is not: the interval does not show a match, and the verdict says
        # which way it leans (negative: the interlock had fewer violations)
        text = self.interlock_text(e(-0.1, -0.2, 0.01))
        assert "matches the shield" not in text and "cannot be separated from the shield" in text
        assert "the estimate leans to the interlock" in text
        text = self.interlock_text(e(0.1, -0.01, 0.2))
        assert "cannot be separated" in text and "the estimate leans to the shield" in text

    def test_a_bound_beyond_five_percent_says_which_is_better(self) -> None:
        worse = self.interlock_text(e(0.2, 0.1, 0.3))  # the interlock has more violations
        assert "is separated from the shield" in worse and "the shield is better" in worse
        better = self.interlock_text(e(-0.2, -0.3, -0.06))  # it has fewer
        assert "is separated from the shield" in better and "the interlock is better" in better

    def test_no_headline_of_a_match_unless_every_pair_matches(self) -> None:
        rows = {
            "H-C": (e(0.0, -0.02, 0.02), 0.9, 0.9, 1.0, 1.0),
            "H-G": (e(-0.1, -0.2, 0.01), 0.9, 0.9, 1.0, 1.0),
        }
        text = " ".join(c_shield(shield_inputs(interlock=rows)).notes)
        assert "headline: the hand-written interlock matches" not in text

    def test_p4_p5_are_reported_with_their_attribution(self) -> None:
        v = c_shield(shield_inputs(p45={"P1": (3, 4)}))
        assert any("P1: P4-P5 violations in 4 episodes, 3 of them attributed" in n for n in v.notes)

    def test_a_missing_r_in_search_is_said(self) -> None:
        v = c_shield(shield_inputs(search_defects=None))
        assert any("R_in search stage did not run" in n for n in v.notes)

    def test_a_missing_cell_is_not_run(self) -> None:
        v = c_shield(shield_inputs(shielded={"H-C1": e(0, 0, 0.01, k=0)}))
        assert v.label == NOT_RUN and "H-G1" in v.headline


# ----------------------------------------------------------------------------- records round trip


def rec(name: str, est: Est, **extra: float | int | str) -> ResultRecord:
    ex = dict(extra)
    if est.k is not None:
        ex["k"] = est.k
    return ResultRecord(
        name=name, value=est.value, n=est.n, fidelity=Fidelity.F1, measured_on="a hand-built one",
        script="tests", ci_lo=est.lo, ci_hi=est.hi, ci_kind="confseq-0.0055", extra=ex,
    )  # fmt: skip


PARAMS = ResultRecord(
    name="run/params", value=1.0, n=1, fidelity=Fidelity.F1, measured_on="params", script="tests",
    extra={"primary_variant": "V3", "monitor_mode": "proxy", "n_max_hostile": 500},
)  # fmt: skip


class TestFromRecords:
    def test_a_claim_without_inputs_is_not_run_and_never_a_quiet_inconclusive(self) -> None:
        verdicts = {v.claim: v for v in verdicts_from_records([PARAMS])}
        assert verdicts["C-CAUSE"].label == NOT_RUN and verdicts["C-SEARCH"].label == NOT_RUN
        assert verdicts["C-TOGETHER"].label == NOT_RUN and verdicts["C-SHIELD"].label == NOT_RUN
        assert "C-PERC.pooled" in verdicts and "C-PERC.trend" in verdicts
        assert all(v.label == NOT_RUN for v in verdicts.values())

    def test_inputs_from_records_give_the_same_verdict_as_the_function(self) -> None:
        records = [
            PARAMS,
            rec(claim_name("C-CAUSE", "q1", "scm-corr"), e(-0.08, -0.12, -0.05, 300)),
            rec(claim_name("C-DIAG", "reversed", "scm-corr"), e(0.1, 0.05, 0.15, 900)),
            rec(claim_name("C-DIAG", "unreversed", "scm-corr"), e(-0.2, -0.25, -0.15, 900)),
        ]
        verdicts = {v.claim: v for v in verdicts_from_records(records)}
        assert verdicts["C-CAUSE"] == c_cause(e(-0.08, -0.12, -0.05, 300))
        assert verdicts["C-DIAG"] == c_diag(e(0.1, 0.05, 0.15, 900), e(-0.2, -0.25, -0.15, 900))

    def test_the_primary_variant_and_the_mode_come_from_the_run_parameters(self) -> None:
        params = dataclasses.replace(
            PARAMS, extra={"primary_variant": "V5", "monitor_mode": "inf", "n_max_hostile": 500}
        )
        records = [params]
        for stratum in ("pooled", "trend"):
            for v in ("V1", "V5"):
                records.append(rec(claim_name("C-PERC", "gap_b", stratum, v), e(0, -0.01, 0.01)))
        for mode in ("opt", "pess", "inf", "proxy"):
            records.append(rec(claim_name("C-MON", "false_alarm", mode), e(0, 0, 0.01, k=0)))
            records.append(rec(claim_name("C-MON", "detection", mode), e(0.6, 0.5, 0.7)))
        by = {v.claim: v for v in verdicts_from_records(records)}
        assert by["C-PERC.pooled"].numbers["primary"] == "V5"
        assert "C-MON" in by and by["C-MON"].headline.startswith("mode inf")
        assert by["C-MON.proxy"].headline.startswith("mode proxy")

    def test_the_nominal_c_plan_verdict_carries_the_shifted_records_of_the_same_arm(self) -> None:
        """0003 A18: built from stored records, the nominal verdict prints the shifted paired
        numbers beside the nominal ones, and the shifted ones decide nothing."""
        records = [PARAMS]
        for condition, cost, fail in (
            ("nominal", e(-0.05, -0.08, -0.03), e(0.0, -0.05, 0.08)),
            ("shifted", e(0.07, 0.01, 0.13), e(0.2, 0.05, 0.35)),
        ):
            records.append(rec(claim_name("C-PLAN", "cost_diff", "P1", condition), cost))
            records.append(rec(claim_name("C-PLAN", "failure_diff", "P1", condition), fail))
        by = {v.claim: v for v in verdicts_from_records(records)}
        nominal, shifted = by["C-PLAN.P1"], by["C-PLAN.P1.shifted"]
        assert nominal.label == WIN
        assert "under shift" in nominal.headline and fmt(0.07) in nominal.headline
        assert nominal.numbers["shifted.cost_diff.hi"] == 0.13
        assert "under shift" not in shifted.headline
        # without the shifted records the nominal verdict is the same, minus the shifted text
        alone = [r for r in records if not r.name.endswith("shifted")]
        only = {v.claim: v for v in verdicts_from_records(alone)}["C-PLAN.P1"]
        assert only.label == WIN and "under shift" not in only.headline

    def test_a_record_without_an_interval_is_not_a_claim_input(self) -> None:
        bare = ResultRecord(
            name="x", value=0.1, n=5, fidelity=Fidelity.F1, measured_on="m", script="s"
        )
        with pytest.raises(ValueError, match="confidence interval"):
            est_of(bare)

    def test_verdict_records_round_trip(self) -> None:
        v = c_shield(shield_inputs(ratios={"H-C": 0.1, "H-G": 0.9, "P": 0.9}))
        back = verdicts_from_stored(verdict_records([v, c_cause(e(0, -0.1, 0.1))]))
        assert back[0].label == WIN and back[0].qualifier == v.qualifier
        assert (back[0].headline, back[0].notes, back[0].rule) == (v.headline, v.notes, v.rule)
        assert back[0].numbers == dict(v.numbers) and back[0].alpha == v.alpha
        assert back[1].claim == "C-CAUSE"

    def test_verdict_records_say_win_loss_and_the_rest_by_sign(self) -> None:
        vs = [c_cause(e(-0.08, -0.12, -0.05)), c_cause(e(0.05, 0.01, 0.09)), c_cause(e(0, -1, 1))]
        assert [r.value for r in verdict_records(vs)] == [1.0, -1.0, 0.0]


def test_the_candidates_of_c_adapt_are_the_registered_ones() -> None:
    assert ADAPT_VARIANTS == ("V2", "V3", "V4", "V5", "V6", "V2L", "V3L")


def test_the_levels_are_the_registered_ones() -> None:
    assert ALPHA_PRIMARY == pytest.approx(0.05 / 9) and ALPHA_PERC == pytest.approx(0.05 / 18)


def test_no_verdict_text_uses_an_absolute() -> None:
    verdicts: list[Verdict] = [
        c_shield(shield_inputs()),
        c_shield(shield_inputs(defects={"P1": 1})),
        c_shield(shield_inputs(ratios={"H-C": 0.1, "H-G": 0.1, "P": 0.1})),
        c_perc("pooled", e(0.06, 0.04, 0.08), e(0, -0.01, 0.01), "V3"),
        c_adapt("V3", e(0.05, 0.03, 0.07), e(0.94, 0.9, 0.97), e(0.95, 0.9, 0.97)),
        c_cause(e(0, -0.1, 0.1)),
        c_diag(e(0.1, 0.05, 0.15), e(0.1, 0.0, 0.2)),
        c_plan("P0", "B3-V3", e(-0.05, -0.08, -0.03), e(0.0, -0.05, 0.08), "nominal"),
        c_mon("proxy", e(0, 0, 0.01, k=0), e(0.6, 0.5, 0.7)),
        c_search(
            SearchPairs(10, 0),
            e(1, 0.7, 1, 10, k=10),
            e(1, 0.7, 1, 10, k=10),
            e(0, -0.3, 0.3, 10),
            None,
            0,
        ),  # fmt: skip
        c_together({"F2-B3": (60, 60)}, (3, 10)),
    ]
    for v in verdicts:
        text = " ".join([v.headline, v.rule, *v.notes])
        assert not FORBIDDEN.search(text), (v.claim, FORBIDDEN.search(text))


class TestSecondaryRowsAreAtTheDescriptiveLevel:
    """Prereg section 6: secondary intervals are shown at 0.05 and are descriptive."""

    def test_the_levels_of_the_verdict_and_of_its_secondary_rows(self) -> None:
        ok = e(0.5, 0.2, 0.8)
        assert c_adapt("V3", ok, ok, ok).alpha == ALPHA_PRIMARY
        assert c_adapt("V5", ok, ok, ok, primary=False).alpha == 0.05
        assert c_mon("proxy", ok, ok, chosen=True).alpha == ALPHA_PRIMARY
        assert c_mon("opt", ok, ok, chosen=False).alpha == 0.05
        assert c_plan("P0", "B3-V3", ok, ok, "nominal").alpha == ALPHA_PRIMARY
        assert c_plan("P0", "B3-V3", ok, ok, "shifted").alpha == 0.05

    def test_the_table_of_verdicts_prints_those_levels(self) -> None:
        ok = e(0.5, 0.2, 0.8)
        from closed_loop.eval.report import _verdict_section

        text = "\n".join(
            _verdict_section([c_adapt("V3", ok, ok, ok), c_adapt("V5", ok, ok, ok, primary=False)])
        )
        assert "0.00556" in text and "0.05000" in text

    def test_a_secondary_row_stands_apart_from_the_registered_claims(self) -> None:
        """A win on a row that decides nothing must not read like a registered win (0003 A20)."""
        from closed_loop.eval.report import SECONDARY_NOTE, _verdict_section, split_verdicts

        ok = e(0.5, 0.2, 0.8)
        claims = [c_adapt("V5", ok, ok, ok, primary=False), c_adapt("V3", ok, ok, ok)]
        registered, secondary = split_verdicts(claims)
        assert [v.alpha for v in registered] == [ALPHA_PRIMARY]
        assert [v.alpha for v in secondary] == [0.05]
        text = "\n".join(_verdict_section(claims))
        before, after = text.split(SECONDARY_NOTE)
        assert "0.00556" in before and "0.05000" not in before
        assert "0.05000" in after

    def test_no_note_without_a_secondary_row(self) -> None:
        from closed_loop.eval.report import SECONDARY_NOTE, _verdict_section

        ok = e(0.5, 0.2, 0.8)
        assert SECONDARY_NOTE not in "\n".join(_verdict_section([c_adapt("V3", ok, ok, ok)]))


class TestFailedEpisodesAreFlagged:
    """0003 A15: a verdict built on cells with failed episodes says so, in the verdict itself."""

    FLAG = "FAILED EPISODES: 3"

    def test_every_claim_function_flags_the_failures_of_its_inputs(self) -> None:
        bad = e(0.5, 0.2, 0.8, failed=3)
        ok = e(0.5, 0.2, 0.8)
        flagged = [
            c_perc("pooled", bad, ok, "V3"),
            c_adapt("V3", bad, ok, ok),
            c_cause(bad),
            c_diag(bad),
            c_plan("P0", "B3-V3", bad, ok, "nominal"),
            c_mon("proxy", bad, ok),
        ]
        for v in flagged:
            assert any(n.startswith(self.FLAG) for n in v.notes), v.claim

    def test_a_clean_run_carries_no_flag(self) -> None:
        ok = e(0.5, 0.2, 0.8)
        assert not any("FAILED" in n for n in c_cause(ok).notes)
        assert not any("FAILED" in n for n in c_mon("proxy", ok, ok).notes)

    def test_the_failures_of_every_input_cell_are_added_up(self) -> None:
        v = c_plan(
            "P0", "B3-V3", e(0.0, -0.1, 0.1, failed=2), e(0.0, -0.1, 0.1, failed=1), "nominal"
        )
        assert any(n.startswith("FAILED EPISODES: 3") for n in v.notes)
