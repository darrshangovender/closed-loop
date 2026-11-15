"""P4c: health sets, the two shield-set sources and the two-truth report. SYNTHETIC data only."""

import math
from functools import cache

import pytest
from hypothesis import given
from hypothesis import strategies as st

from closed_loop.core.types import Fidelity, bin_of_rul
from closed_loop.perception.conformal import ALPHA_LIFE, development_data
from closed_loop.perception.contract import (
    ALPHA_STEP,
    SHIELD_SETS,
    ContractCounts,
    ContractRow,
    bin_in_set,
    contract_counts,
    contract_report,
    contract_row,
    health_set,
    make_shield_provider,
    nominal_miss_level,
    report_records,
    shield_set_alpha,
)
from closed_loop.perception.estimator import TheilSenEstimator
from closed_loop.perception.labels import Checkpoint, RulInterval
from closed_loop.plant import F1Plant, Scenario
from closed_loop.plant.params import ASSET_IDS, PlantParams

P = PlantParams()
INF = math.inf


def iv(lo, m, hi):
    return RulInterval(lo, m, hi, 0.9, False)


class TestHealthSet:
    @pytest.mark.parametrize(
        ("lo", "m", "hi", "want"),
        [
            (35.0, 40.0, 60.0, (0, 0)),  # all above 30 days
            (12.0, 15.0, 25.0, (1, 1)),
            (5.0, 8.0, 12.0, (1, 2)),  # 12 is bin 1, 5 is bin 2
            (2.0, 4.0, 40.0, (0, 3)),  # spans the whole range
            (-5.0, 1.0, 2.0, (3, 3)),  # a negative lower bound is the critical bin
            (3.0, 5.0, 10.0, (2, 2)),  # the edges: exactly 10 and exactly 3 are bin 2
            (30.0, 30.0, 30.0, (1, 1)),  # exactly 30 is bin 1
            (-INF, 20.0, INF, (0, 3)),  # an interval wider than the range clips to (0, 3)
            (1.0, 2.0, INF, (0, 3)),
            (50.0, 60.0, INF, (0, 0)),
        ],
    )
    def test_cases(self, lo, m, hi, want):
        assert health_set(iv(lo, m, hi)) == want

    def test_the_point_estimate_is_always_inside_the_set(self):
        # an adaptive method that narrowed the interval past the point estimate
        assert health_set(iv(20.0, 2.0, 25.0)) == (1, 3)
        assert health_set(iv(1.0, 50.0, 2.0)) == (0, 3)

    @given(
        st.floats(-20, 200, allow_nan=False),
        st.floats(0, 200, allow_nan=False),
        st.floats(0, 100, allow_nan=False),
    )
    def test_sets_always_contain_the_point_estimate_bin_and_the_interval(self, lo, m, width):
        interval = iv(lo, m, lo + width)
        low, high = health_set(interval)
        assert 0 <= low <= high <= 3
        assert low <= bin_of_rul(m) <= high
        assert low <= bin_of_rul(lo + width) and bin_of_rul(lo) <= high

    def test_bin_in_set(self):
        assert bin_in_set(2, (1, 3)) and bin_in_set(1, (1, 3)) and bin_in_set(3, (1, 3))
        assert not bin_in_set(0, (1, 3))
        assert not bin_in_set(3, (0, 2))


@cache
def _dev():
    return development_data(P, range(2000, 2030))


class TestSources:
    def test_alphas_and_levels(self):
        assert SHIELD_SETS == ("step99", "life95")
        assert shield_set_alpha("step99") == ALPHA_STEP == 0.01
        assert shield_set_alpha("life95") == ALPHA_LIFE == 0.05
        assert nominal_miss_level("step99") == 0.01
        assert nominal_miss_level("life95") == 0.05
        with pytest.raises(ValueError, match="shield_set"):
            shield_set_alpha("step95")

    def test_step99_is_the_arms_variant_at_alpha_one_percent(self):
        for variant in ("V1", "V3", "V5"):
            p = make_shield_provider("step99", variant=variant, development=_dev(), seed=1)
            assert p.name == variant and p.state()["alpha"] == 0.01

    def test_life95_is_the_v6_band_whatever_the_variant(self):
        for variant in ("V1", "V3", "V5"):
            p = make_shield_provider("life95", variant=variant, development=_dev(), seed=1)
            assert p.name == "V6" and p.state()["alpha"] == 0.05

    def test_unknown_source(self):
        with pytest.raises(ValueError, match="shield_set"):
            make_shield_provider("nope", variant="V1", development=_dev(), seed=1)

    def test_the_per_life_band_is_wider_than_the_step_set(self):
        from closed_loop.perception.estimator import EstimatorOutput

        out = EstimatorOutput("PU-201", 40, "estimated", 25.0, 5.0, 4.0)
        step = make_shield_provider("step99", variant="V1", development=_dev(), seed=1)
        life = make_shield_provider("life95", variant="V1", development=_dev(), seed=1)
        a, b = step.predict(out, 0.99), life.predict(out, 0.95)
        # a band that holds all of a life's ~60 checkpoints at 95% is wider than a per-step 99% one
        assert b.hi - b.lo > a.hi - a.lo


def cp(day, lo, hi, true, bin_, *, life=0, asset="PU-201", est="estimated", m=None):
    m = 0.5 * (lo + hi) if m is None else m
    return Checkpoint(life, asset, day, lo, m, hi, 3.0, 0.95, est, true_rul=true, health_bin=bin_)


class TestCounts:
    def test_the_two_readings_can_disagree_and_that_is_the_finding(self):
        """A10: the realised RUL is inside the interval, the rate-RUL bin is not in the set."""
        # interval [40, 60] (bin 0 only), realised RUL 45 is inside, but a load step moved the
        # rate RUL to 5 days: bin 2
        c = contract_counts([cp(1, 40.0, 60.0, 45.0, 2)])
        assert (c.lives, c.lives_miss_rul, c.lives_miss_bin) == (1, 0, 1)
        row = ContractRow("life95", c)
        assert row.any_miss_rul == 0.0 and row.any_miss_bin == 1.0 and row.gap == 1.0

    def test_the_reverse_disagreement(self):
        # realised RUL 70 is outside [40, 60] but the bin (0) is inside the set (0, 0)
        c = contract_counts([cp(1, 40.0, 60.0, 70.0, 0)])
        assert (c.lives_miss_rul, c.lives_miss_bin) == (1, 0)

    def test_per_life_and_per_checkpoint_counts(self):
        cps = [
            cp(1, 40.0, 60.0, 45.0, 0, life=0),
            cp(2, 40.0, 60.0, 70.0, 0, life=0),  # life 0: a coverage miss
            cp(1, 5.0, 12.0, 8.0, 2, life=1),
            cp(2, 5.0, 12.0, 6.0, 2, life=1),  # life 1: clean
            cp(1, 5.0, 12.0, 8.0, 3, life=0, asset="PU-202"),  # a different asset's life 0
        ]
        c = contract_counts(cps)
        assert c.lives == 3 and c.lives_miss_rul == 1 and c.lives_miss_bin == 1
        assert c.checkpoints == 5 and c.miss_rul == 1 and c.miss_bin == 1
        # widths: (0,0)=1, (0,0)=1, (1,2)=2, (1,2)=2, (1,2)=2
        assert c.width_sum == 8
        row = ContractRow("step99", c)
        assert row.mean_width == pytest.approx(8 / 5)
        assert row.step_miss_rul == pytest.approx(1 / 5)
        assert row.any_miss_rul == pytest.approx(1 / 3)

    def test_populations(self):
        cps = [cp(1, 40.0, 60.0, 90.0, 0, est="prior"), cp(2, 40.0, 60.0, 45.0, 0)]
        assert contract_counts(cps, "estimated").lives_miss_rul == 0
        assert contract_counts(cps, "prior").lives_miss_rul == 1
        assert contract_counts(cps, "all").lives_miss_rul == 1

    def test_a_checkpoint_without_both_truths_is_refused(self):
        bare = Checkpoint(0, "PU-201", 1, 1.0, 2.0, 3.0, 1.0, 0.95)
        with pytest.raises(ValueError, match="both truths"):
            contract_counts([bare])
        half = Checkpoint(0, "PU-201", 1, 1.0, 2.0, 3.0, 1.0, 0.95, true_rul=2.0)
        with pytest.raises(ValueError, match="both truths"):
            contract_counts([half])

    def test_counts_add_and_the_report_accepts_pooled_counts(self):
        a = contract_counts([cp(1, 40.0, 60.0, 45.0, 2)])
        b = contract_counts([cp(1, 40.0, 60.0, 45.0, 0), cp(2, 40.0, 60.0, 90.0, 0)])
        pooled = a + b
        assert pooled == ContractCounts(2, 1, 1, 3, 1, 1, 3)
        rep = contract_report(pooled, a)
        assert rep["step99"].counts == pooled and rep["life95"].counts == a
        assert contract_report([cp(1, 40.0, 60.0, 45.0, 2)], pooled)["step99"].counts == a

    def test_empty_rows_have_nan_rates(self):
        row = contract_row("step99", [])
        assert row.counts == ContractCounts()
        assert math.isnan(row.any_miss_rul) and math.isnan(row.mean_width)

    def test_bad_source(self):
        with pytest.raises(ValueError, match="shield_set"):
            contract_row("x", [])


class TestRecords:
    def test_records_carry_wilson_intervals_and_both_readings(self):
        counts = ContractCounts(40, 4, 10, 800, 12, 30, 1000)
        recs = report_records(
            contract_report(counts, counts),
            measured_on="synthetic fixture",
            script="tests.perception.test_contract",
            fidelity=Fidelity.F1,
        )
        names = {r.name for r in recs}
        assert names == {
            f"contract.{s}.any_miss_{r}" for s in ("step99", "life95") for r in ("rul", "bin")
        }
        by = {r.name: r for r in recs}
        assert by["contract.step99.any_miss_rul"].value == pytest.approx(0.1)
        assert by["contract.step99.any_miss_bin"].value == pytest.approx(0.25)
        for r in recs:
            assert r.n == 40 and r.ci_lo is not None and r.ci_hi is not None
            assert r.ci_lo <= r.value <= r.ci_hi and r.ci_kind == "wilson-0.95"
            assert r.fidelity is Fidelity.F1

    def test_a_source_with_no_life_has_no_records(self):
        recs = report_records(
            contract_report(ContractCounts(), ContractCounts(10, 1, 1, 100, 2, 2, 100)),
            measured_on="x",
            script="y",
            fidelity=Fidelity.F1,
        )
        assert {r.name.split(".")[1] for r in recs} == {"life95"}


class TestOnThePlant:
    """The wiring from the plant's truth to the report, on open-loop episodes (nobody acts)."""

    def episode(self, seed):
        plant = F1Plant(P)
        obs = plant.reset(seed, Scenario())
        est = {a: TheilSenEstimator(P) for a in ASSET_IDS}
        step = make_shield_provider("step99", variant="V1", development=_dev(), seed=seed)
        life = make_shield_provider("life95", variant="V1", development=_dev(), seed=seed)
        cps = {"step99": [], "life95": []}
        for day in range(P.horizon_days):
            truth = plant.truth
            for a in ASSET_IDS:
                if truth.true_rul_days[a] <= 0.0:
                    continue
                hi = obs.assets[a].hi_last
                out = est[a].update(a, day, hi, truth.life_id[a])
                for name, prov in (("step99", step), ("life95", life)):
                    level = 1.0 - shield_set_alpha(name)
                    r = prov.predict(out, level)
                    cps[name].append(
                        Checkpoint(
                            truth.life_id[a],
                            a,
                            day,
                            r.lo,
                            r.m,
                            r.hi,
                            out.sigma_days,
                            level,
                            out.status,
                            true_rul=truth.true_rul_days[a],
                            health_bin=truth.health_bin[a],
                        )
                    )
            for _ in range(96):
                obs = plant.step([]).obs
        return cps

    def test_the_report_runs_on_plant_truth_and_the_counts_are_consistent(self):
        total = {"step99": ContractCounts(), "life95": ContractCounts()}
        n_cp = 0
        for seed in range(1000, 1004):
            cps = self.episode(seed)
            n_cp += len(cps["step99"])
            for name in total:
                total[name] = total[name] + contract_counts(cps[name])
        rep = contract_report(total["step99"], total["life95"])
        assert len(cps["step99"]) == len(cps["life95"])
        assert rep["step99"].counts.checkpoints == n_cp == rep["life95"].counts.checkpoints
        assert rep["step99"].counts.lives == rep["life95"].counts.lives == 16
        for row in rep.values():
            assert 0.0 <= row.any_miss_rul <= 1.0 and 0.0 <= row.any_miss_bin <= 1.0
            assert 0.0 <= row.step_miss_rul <= 1.0 and 0.0 <= row.step_miss_bin <= 1.0
            assert 1.0 <= row.mean_width <= 4.0
        # the per-life band is wider than the per-step set
        assert rep["life95"].mean_width >= rep["step99"].mean_width
