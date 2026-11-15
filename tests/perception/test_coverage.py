"""P4a: coverage metrics. SYNTHETIC data only."""

import math
from functools import cache

import numpy as np
import pytest

from closed_loop.perception.coverage import (
    CoverageRun,
    any_miss_rate,
    checkpoint_coverage,
    gaps,
    learner_coverage,
    manski_bounds,
    open_loop_lives,
    oracle_coverage,
    oracle_lower_coverage,
    per_life_any_miss,
    per_life_any_miss_labels,
    per_life_counts,
    select,
)
from closed_loop.perception.labels import (
    Checkpoint,
    Label,
    LabelStatus,
    RulInterval,
    resolve_labels,
)
from closed_loop.plant.params import ASSET_IDS, PlantParams

H, M, LH, U = LabelStatus.HIT, LabelStatus.MISS, LabelStatus.LOWER_HIT, LabelStatus.UNKNOWN
P = PlantParams()


@cache
def _lives(seed):
    return open_loop_lives(P, seed)


def cp(day, lo, hi, true, *, life=0, asset="PU-201", est="estimated", **kw):
    return Checkpoint(life, asset, day, lo, 0.5 * (lo + hi), hi, 3.0, 0.9, est, true_rul=true, **kw)


def lab(
    status,
    *,
    day=10,
    lo=5.0,
    hi=20.0,
    end=30.0,
    censored=True,
    life=0,
    asset="PU-201",
    prior=False,
    blind=False,
):
    return Label(
        life,
        asset,
        day,
        lo,
        hi,
        0.9,
        math.ceil(end),
        status,
        end,
        censored,
        False,
        prior,
        blind=blind,
    )


class TestOracle:
    def test_hand_case_with_edges(self):
        ivs = np.array([[5.0, 20.0], [5.0, 20.0], [5.0, 20.0], [5.0, 20.0]])
        truth = [5.0, 20.0, 4.99, 20.01]
        assert oracle_coverage(ivs, truth) == 0.5
        assert oracle_lower_coverage(ivs, truth) == 0.75  # only 4.99 is below lo

    def test_intervals_may_be_objects(self):
        ivs = [
            RulInterval(5.0, 10.0, 20.0, 0.9, False),
            RulInterval(-math.inf, 3.0, 4.0, 0.9, True),
        ]
        assert oracle_coverage(ivs, [12.0, 9.0]) == 0.5

    def test_mask_keeps_a_subset_and_empty_is_nan(self):
        ivs = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        truth = [0.5, 3.0, 0.5]
        assert oracle_coverage(ivs, truth, mask=[True, True, False]) == 0.5
        assert math.isnan(oracle_coverage(ivs, truth, mask=[False, False, False]))
        assert math.isnan(oracle_lower_coverage(np.empty((0, 2)), []))

    def test_bad_inputs(self):
        ivs = np.array([[0.0, 1.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="NaN"):
            oracle_coverage(ivs, [0.5, math.nan])
        with pytest.raises(ValueError, match="true RUL"):
            oracle_coverage(ivs, [0.5])
        with pytest.raises(ValueError, match="mask"):
            oracle_coverage(ivs, [0.5, 0.5], mask=[True])
        with pytest.raises(ValueError, match="shape"):
            oracle_coverage(np.zeros((3, 3)), [0.0, 0.0, 0.0])

    def test_checkpoint_coverage_selects_a_population_and_needs_truth(self):
        cps = [
            cp(1, 0, 5, 3.0),
            cp(2, 0, 5, 9.0, est="insufficient"),
            cp(3, 0, 5, 9.0, est="prior"),
        ]
        assert checkpoint_coverage(cps, "all") == pytest.approx(1 / 3)
        assert checkpoint_coverage(cps, "estimated") == 1.0
        assert checkpoint_coverage(cps, "prior") == 0.0
        assert checkpoint_coverage(cps, "post_blind") == 0.5  # drops only the blind one
        assert math.isnan(checkpoint_coverage([], "all"))
        no_truth = Checkpoint(0, "PU-201", 1, 0.0, 1.0, 2.0, 1.0, 0.9)
        with pytest.raises(ValueError, match="truth"):
            checkpoint_coverage([no_truth])


class TestPopulations:
    def test_select_follows_the_prior_and_blind_flags(self):
        labels = [lab(H), lab(H, prior=True), lab(H, prior=True, blind=True)]
        assert len(select(labels, "all")) == 3
        assert len(select(labels, "estimated")) == 1
        assert len(select(labels, "prior")) == 2
        assert len(select(labels, "post_blind")) == 2
        with pytest.raises(ValueError, match="population"):
            select(labels, "everything")


class TestLearnerCoverage:
    def test_counts_rates_and_the_two_estimates(self):
        labels = [
            lab(H, end=22.0, censored=False),  # lo 5 held (22 >= 15)
            lab(H, end=22.0, censored=False),
            lab(M, end=12.0, censored=False),  # failure before t + lo: lower bound failed
            lab(M, end=40.0, censored=True),  # censored past t + hi: lower held, upper failed
            lab(LH, end=25.0),
            lab(LH, end=15.0),  # exactly t + lo: the lower bound held
            lab(U, end=12.0),
            lab(U, end=12.0),
            lab(U, end=11.0),
        ]
        c = learner_coverage(labels)
        assert (c["n"], c["hit"], c["miss"], c["lower_hit"], c["unknown"]) == (9, 2, 2, 2, 3)
        assert c["informative"] == 6
        assert c["c_lrn"] == pytest.approx(2 / 6)  # HIT over HIT + MISS + LOWER_HIT
        assert c["c_lrn_lo"] == pytest.approx(5 / 6)  # one failure fell below its lower bound
        assert c["hit_rate"] == pytest.approx(2 / 9) and c["unknown_rate"] == pytest.approx(3 / 9)

    def test_unknown_is_never_a_hit_or_a_miss(self):
        c = learner_coverage([lab(U), lab(U)])
        assert c["informative"] == 0 and math.isnan(c["c_lrn"]) and math.isnan(c["c_lrn_lo"])
        assert c["unknown_rate"] == 1.0

    def test_empty_population(self):
        c = learner_coverage([])
        assert c["n"] == 0 and math.isnan(c["c_lrn"]) and math.isnan(c["hit_rate"])

    def test_population_filter(self):
        labels = [lab(H, end=22.0, censored=False), lab(M, end=12.0, censored=False, prior=True)]
        assert learner_coverage(labels, "estimated")["c_lrn"] == 1.0
        assert learner_coverage(labels, "prior")["c_lrn"] == 0.0


class TestManski:
    def test_bounds_on_a_hand_built_set(self):
        labels = [lab(H), lab(H), lab(M), lab(LH), lab(U)]
        lo, hi = manski_bounds(labels)
        assert lo == pytest.approx(2 / 5)  # only the HITs are certain hits
        assert hi == pytest.approx(4 / 5)  # everything but the one MISS might be a hit

    def test_lower_hit_widens_the_bounds_because_its_upper_side_is_unknown(self):
        assert manski_bounds([lab(LH)]) == (0.0, 1.0)
        assert manski_bounds([lab(U)]) == (0.0, 1.0)
        assert manski_bounds([lab(H)]) == (1.0, 1.0)
        assert manski_bounds([lab(M)]) == (0.0, 0.0)

    def test_no_labels_means_no_information(self):
        assert manski_bounds([]) == (0.0, 1.0)

    def test_population(self):
        labels = [lab(H), lab(M, prior=True)]
        assert manski_bounds(labels, "estimated") == (1.0, 1.0)
        assert manski_bounds(labels, "prior") == (0.0, 0.0)


def _closed_loop_run(seed, k, scale):
    """First lives of the four assets under a bound rule: replace one day before the lower bound
    falls below ``k`` days (a replacement that renews nothing here: the run only needs the labels).
    Intervals are ``m +- scale * sigma``, an arbitrary rule; the bracket must hold for any rule."""
    cps_all: list[Checkpoint] = []
    labels_all: list[Label] = []
    for life in _lives(seed):
        cps: list[Checkpoint] = []
        tau: float | None = None
        for d, out, truth in zip(life.days, life.outputs, life.true_rul, strict=True):
            lo = max(out.m_days - scale * out.sigma_days, 0.0)
            hi = out.m_days + scale * out.sigma_days
            cps.append(
                Checkpoint(
                    0,
                    life.asset_id,
                    int(d),
                    lo,
                    out.m_days,
                    hi,
                    out.sigma_days,
                    0.9,
                    out.status,
                    true_rul=float(truth),
                )
            )
            if out.status == "estimated" and lo < k:
                tau = float(d) + 1.0
                break
        if tau is None or tau >= life.end_day:  # not replaced before it ended on its own
            end, censored = life.end_day, life.censored
        else:
            end, censored = tau, True
        cps = [c for c in cps if c.day <= end]
        cps_all += cps
        labels_all += resolve_labels(cps, end, censored, audit=False)
    return cps_all, labels_all


class TestManskiBracketsTheOracleOnSimulatedRuns:
    """Done-when of P4a: in every simulated run the Manski bounds bracket the oracle coverage."""

    @pytest.mark.parametrize("k", [2.0, 7.0, 14.0, 1e9])
    @pytest.mark.parametrize("scale", [0.5, 1.5, 3.0])
    def test_bracket_per_seed_and_population(self, k, scale):
        n_with_unknown = 0
        for seed in range(1000, 1012):
            cps, labels = _closed_loop_run(seed, k, scale)
            for pop in ("all", "estimated", "prior", "post_blind"):
                sel_c, sel_l = select(cps, pop), select(labels, pop)
                if not sel_c:
                    continue
                orc = checkpoint_coverage(cps, pop)
                lo, hi = manski_bounds(labels, pop)
                assert lo - 1e-12 <= orc <= hi + 1e-12, (seed, pop)
                assert len(sel_c) == len(sel_l)
                n_with_unknown += any(x.status is U for x in sel_l)
        assert n_with_unknown > 0 or k > 1e6  # the bound rule really produces unknowns

    def test_lower_bound_coverage_of_the_learner_matches_the_oracle_where_nothing_is_unknown(self):
        cps, labels = _closed_loop_run(1003, 1e9, 1.5)  # nobody is replaced: failures or horizon
        c = learner_coverage(labels)
        orc_lo = checkpoint_coverage(cps, "all", lower=True)
        informative = [x for x in labels if x.status is not U]
        # the learner's lower-bound coverage on the informative labels is exact for those labels
        truth = {(x.asset_id, x.checkpoint_day): x.true_rul for x in labels}
        exact = np.mean([truth[(x.asset_id, x.checkpoint_day)] >= x.lo for x in informative])
        assert c["c_lrn_lo"] == pytest.approx(exact)
        assert 0.0 <= orc_lo <= 1.0

    def test_censoring_before_the_lower_bound_leaves_unknowns_and_wide_bounds(self):
        cps, labels = _closed_loop_run(1004, 7.0, 1.5)
        lo, hi = manski_bounds(labels)
        assert hi - lo > 0.5  # most labels say nothing about both sides


class TestPerLifeAnyMiss:
    def test_hand_case(self):
        cps = [
            cp(1, 0, 5, 3.0, life=0),
            cp(2, 0, 5, 9.0, life=0),  # life 0: one miss
            cp(1, 0, 5, 3.0, life=1),
            cp(2, 0, 5, 4.0, life=1),  # life 1: none
            cp(1, 0, 5, 9.0, life=0, asset="PU-202"),  # a different asset's life 0: a miss
        ]
        assert per_life_counts(cps) == (2, 3)
        assert per_life_any_miss(cps) == pytest.approx(2 / 3)

    def test_populations_and_empty(self):
        cps = [cp(1, 0, 5, 9.0, est="prior"), cp(2, 0, 5, 3.0)]
        assert per_life_any_miss(cps, "estimated") == 0.0
        assert per_life_any_miss(cps, "prior") == 1.0
        assert math.isnan(per_life_any_miss([], "all"))

    def test_the_per_life_rate_is_at_least_the_per_checkpoint_miss_rate(self):
        cps, _ = _closed_loop_run(1005, 1e9, 1.0)
        miss = 1.0 - checkpoint_coverage(cps)
        assert per_life_any_miss(cps) >= miss - 1e-12

    def test_any_miss_rate_generic(self):
        assert any_miss_rate(["a", "a", "b", "c"], [False, True, False, False]) == pytest.approx(
            1 / 3
        )
        assert math.isnan(any_miss_rate([], []))
        with pytest.raises(ValueError, match="same length"):
            any_miss_rate(["a"], [True, False])

    def test_learner_version_counts_miss_labels_only(self):
        labels = [lab(H, life=0), lab(M, life=0), lab(U, life=1), lab(LH, life=1), lab(M, life=2)]
        assert per_life_any_miss_labels(labels) == pytest.approx(2 / 3)


class TestGaps:
    def _runs(self):
        open_cps = [cp(d, 0, 10, 5.0, life=0) for d in range(10)]  # all covered
        closed_cps = [cp(d, 0, 10, 5.0 if d < 5 else 50.0) for d in range(10)]  # half covered
        closed_labels = [
            lab(H, end=15.0, censored=False),
            lab(H, end=15.0, censored=False),
            lab(M, end=12.0, censored=False),  # a failure before its lower bound
            lab(U, end=11.0),
        ]
        return CoverageRun(open_cps), CoverageRun(closed_cps, closed_labels)

    def test_gap_a_and_gap_b_on_a_hand_case(self):
        open_run, closed_run = self._runs()
        g = gaps(open_run, closed_run)
        assert g["c_orc_open"] == 1.0 and g["c_orc_closed"] == 0.5
        assert g["c_lrn_closed"] == pytest.approx(2 / 3)
        assert g["gap_a"] == pytest.approx(2 / 3 - 0.5)  # C_lrn - C_orc
        assert g["gap_b"] == pytest.approx(0.5)  # C_orc(open) - C_orc(closed)
        assert g["manski_lo"] == pytest.approx(2 / 4) and g["manski_hi"] == pytest.approx(3 / 4)
        assert g["unknown_rate"] == pytest.approx(1 / 4)

    def test_lower_bound_versions(self):
        open_run, closed_run = self._runs()
        g = gaps(open_run, closed_run)
        assert g["c_orc_lo_open"] == 1.0 and g["c_orc_lo_closed"] == 1.0
        assert g["gap_b_lo"] == 0.0
        assert g["c_lrn_lo_closed"] == pytest.approx(2 / 3)
        assert g["gap_a_lo"] == pytest.approx(2 / 3 - 1.0)

    def test_a_population_with_nothing_in_it_is_an_error(self):
        open_run, closed_run = self._runs()
        with pytest.raises(ValueError, match="no checkpoints"):
            gaps(open_run, closed_run, population="prior")
        no_info = CoverageRun(closed_run.checkpoints, [lab(U)])
        with pytest.raises(ValueError, match="informative"):
            gaps(open_run, no_info)


class TestOpenLoopLives:
    def test_structure(self):
        lives = open_loop_lives(P, 1000)
        assert [x.asset_id for x in lives] == list(ASSET_IDS)
        for x in lives:
            assert len(x.days) == len(x.outputs) == len(x.true_rul) > 0
            assert list(x.days) == list(range(len(x.days)))
            assert all(x.days < x.failure_day) and all(x.true_rul > 0.0)
            assert x.true_rul[0] == pytest.approx(x.failure_day, abs=0.05)
            assert x.censored == (x.failure_day >= x.horizon_days)
            assert x.end_day == (x.horizon_days if x.censored else x.failure_day)

    def test_deterministic(self):
        a, b = open_loop_lives(P, 1001), open_loop_lives(P, 1001)
        assert [x.outputs for x in a] == [x.outputs for x in b]

    def test_about_sixty_percent_fail_inside_the_horizon(self):
        n = fails = 0
        for seed in range(1000, 1100):
            for x in open_loop_lives(P, seed):
                n += 1
                fails += not x.censored
        assert 0.5 < fails / n < 0.7
