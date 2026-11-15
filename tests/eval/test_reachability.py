"""P12c (0003 A16): can each verdict be reached at its registered size?

Every claim's rule is run on made-up per-seed values at the registered count, through the code the
benchmark uses, on a perfect input and on a useless one. A case that the registered rules cannot
satisfy is an ``xfail(strict=True)`` with the arithmetic as its reason, so the day the numbers or
the rules change it turns up as an unexpected pass. All of it is made-up data: it tests the
statistics and the rules, never the plant.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from closed_loop.eval import reachability as R
from closed_loop.eval import stages
from closed_loop.eval.claims import INCONCLUSIVE, LOSS, WIN
from closed_loop.eval.counts import ALPHA_PERC, ALPHA_PRIMARY, FULL
from closed_loop.eval.report import FORBIDDEN
from closed_loop.eval.runner import Runner
from closed_loop.eval.stages import StageContext, run_pipeline
from closed_loop.eval.tuned import PROVISIONAL
from closed_loop.stats.confseq import _spread_halfwidth


def case_params() -> list[Any]:
    out = []
    for case in R.CASES:
        marks = [] if case.reachable else [pytest.mark.xfail(strict=True, reason=case.why)]
        out.append(pytest.param(case, id=case.id, marks=marks))
    return out


class TestEveryRuleIsReachable:
    @pytest.mark.parametrize("case", case_params())
    def test_the_registered_rule_gives_the_input_its_label(self, case: R.Case) -> None:
        verdict = case.run(1.0)
        assert verdict.label == case.want, f"{case.what}: {case.bound(verdict)}"

    def test_every_claim_with_a_win_and_a_loss_is_tried_on_a_perfect_and_a_useless_input(
        self,
    ) -> None:
        have = {(c.claim, c.leg, c.kind) for c in R.CASES}
        for claim, leg in (
            ("C-PERC.pooled", "pooled"),
            ("C-PERC.trend", "trend"),
            ("C-ADAPT", ""),
            ("C-CAUSE", ""),
            ("C-DIAG", ""),
            ("C-PLAN.P0", "cost"),
            ("C-PLAN.P0", "safety"),
            ("C-MON", ""),
            ("C-SHIELD", ""),
            ("C-SEARCH", ""),
        ):
            assert (claim, leg, "perfect") in have, (claim, leg)
            assert (claim, leg, "useless") in have, (claim, leg)
        wants = {(c.claim, c.leg, c.kind): c.want for c in R.CASES}
        assert all(wants[k] == WIN for k in have if k[2] == "perfect")
        assert all(wants[k] == LOSS for k in have if k[2] == "useless")

    def test_the_ids_are_unique(self) -> None:
        assert len({c.id for c in R.CASES}) == len(R.CASES)

    def test_the_cases_that_cannot_reach_their_label_are_exactly_two_and_say_why(self) -> None:
        cannot = [c.id for c in R.CASES if not c.reachable]
        assert cannot == ["C-PERC.trend/trend/useless", "C-PLAN.P0/cost/perfect"]
        for c in R.CASES:
            assert bool(c.why) == (not c.reachable), c.id

    def test_a_case_marked_unreachable_really_does_not_reach_its_label(self) -> None:
        for c in R.CASES:
            if not c.reachable:
                assert c.run(1.0).label != c.want, c.id

    def test_the_sizes_are_the_registered_counts(self) -> None:
        n = {c.id: c.n for c in R.CASES}
        assert n["C-PERC.pooled/pooled/perfect"] == FULL.open_loop_seeds == 500
        assert n["C-ADAPT/perfect"] == FULL.n_max_perception == 500
        assert n["C-PLAN.P0/cost/perfect"] == FULL.n_max_planner == 300
        assert n["C-MON/perfect"] == FULL.mon_nominal == 700
        assert n["C-MON/useless"] == FULL.mon_shifted == 300
        assert n["C-SHIELD/perfect"] == FULL.n_max_hostile == 500
        assert n["C-SEARCH/perfect"] == FULL.search_runs == 10
        assert n["C-CAUSE/perfect"] == round(FULL.causal_test_seeds * 4 * 0.6) == 180
        assert n["C-DIAG/perfect"] == round(FULL.mdiag_test_seeds * 4 * 0.6) == 720

    def test_the_sizes_follow_the_counts_they_are_built_from(self) -> None:
        small = dataclasses.replace(FULL, n_max_planner=150, search_runs=8, mon_nominal=400)
        by = {c.id: c.n for c in R._build_cases(small)}
        assert by["C-PLAN.P0/cost/perfect"] == 150
        assert by["C-SEARCH/perfect"] == 8
        assert by["C-MON/perfect"] == 400

    def test_a_planner_cost_leg_does_win_when_the_saving_is_large_enough(self) -> None:
        # the same code, a saving of 0.5 of the cap: the cost leg is not the obstacle in principle
        v = R._plan(None, 0.0, 0.0, 1.0, FULL)["C-PLAN.P0"]
        assert v.label == WIN and v.numbers["cost_diff.hi"] < -0.02


class TestWhatCannotBeReached:
    def test_the_trend_stratum_loss_needs_a_quieter_gap_than_the_one_measured_on_dev(self) -> None:
        case = next(c for c in R.CASES if c.id == "C-PERC.trend/trend/useless")
        factor = _spread_halfwidth(500, 1.0, ALPHA_PERC)
        assert factor * R.SPREADS.gap_b_trend_sd > 0.03  # 0.1765 x 0.22 = 0.039
        assert factor * R.SPREADS.gap_b_sd < 0.03  # the pooled stratum is under it
        assert case.run(1.0).label == INCONCLUSIVE
        # a spread of at most 0.03 / 0.1765 = 0.17 per seed is necessary (the sample gap is not
        # exactly 0 either); at 0.4 of the stated 0.22 the loss is there
        assert case.run(0.4).label == LOSS
        assert "0.170" in case.why and "0.039" in case.why

    def test_the_cost_leg_arithmetic_of_the_planner(self) -> None:
        mean, sd = R._b3_cost_moments(R.SPREADS)
        # B3 once it no longer stalls (0003 A17, A18): 5.33 units, sd 0.89 on dev (8.83, 17.8 once)
        assert (mean, sd) == pytest.approx((5.33, 0.89), abs=0.02)
        factor = _spread_halfwidth(300, 1.0, ALPHA_PRIMARY)
        saving = mean - R.CHEAPEST_PLAN_UNITS
        assert saving < 4.0 + factor * sd  # 3.33 < 4.19: out of reach even on the spread bound
        assert saving < 4.0  # and under the margin before any half-width is added
        floor_units = R.floors(sizes=(300,))[0].betting * 200.0
        assert floor_units == pytest.approx(16.0, abs=0.1)
        assert saving < 4.0 + floor_units  # and far out of reach on the betting one
        case = next(c for c in R.CASES if c.id == "C-PLAN.P0/cost/perfect")
        for text in ("kurtosis", "16 units", "4.19", "3.33", "2 (two planned jobs"):
            assert text in case.why, text

    def test_the_cost_of_b3_no_longer_has_a_tail_and_the_cheapest_plan_still_falls_short(
        self,
    ) -> None:
        rng = np.random.default_rng(1)
        x = R._b3_cost(rng, 200_000, R.SPREADS)
        mean, sd = R._b3_cost_moments(R.SPREADS)
        assert x.mean() == pytest.approx(mean, rel=0.02) and x.std() == pytest.approx(sd, rel=0.02)
        d = x - R.CHEAPEST_PLAN_UNITS
        kurt = float(np.mean((d - d.mean()) ** 4) / np.var(d) ** 2)
        assert (
            kurt < 300 / 50
        )  # B3 alone passes the guard now (before: 35 on dev): the spread bound
        # ... and the cheapest honest plan misses the margin on it, by less than a unit
        v = R._plan(R.CHEAPEST_PLAN_UNITS, 0.0, 0.0, 1.0, FULL)["C-PLAN.P0"]
        assert v.label == INCONCLUSIVE
        assert -0.02 < v.numbers["cost_diff.hi"] < -0.015


class TestTheMadeUpDataAreWhatTheyClaim:
    def test_a_clipped_normal_has_the_mean_and_sd_asked_for(self) -> None:
        rng = np.random.default_rng(2)
        for mean, sd in ((0.95, 0.075), (0.88, 0.07), (0.5, 0.2)):
            x = R._bounded(rng, 400_000, mean, sd)
            assert x.min() >= 0.0 and x.max() <= 1.0
            assert x.mean() == pytest.approx(mean, abs=2e-3) and x.std() == pytest.approx(
                sd, abs=2e-3
            )

    def test_an_impossible_mean_and_sd_are_refused(self) -> None:
        with pytest.raises(ValueError, match="no clipped normal"):
            R.clipped_normal_latent(1.0, 0.05)

    def test_a_pair_has_the_means_the_sds_and_the_correlation_asked_for(self) -> None:
        rng = np.random.default_rng(3)
        a, b = R._bounded_pair(rng, 300_000, 0.9, 0.6, 0.1, 0.8)
        assert a.mean() == pytest.approx(0.9, abs=2e-3) and b.mean() == pytest.approx(0.6, abs=2e-3)
        assert a.std() == pytest.approx(0.1, abs=2e-3) and b.std() == pytest.approx(0.1, abs=2e-3)
        assert 0.7 < float(np.corrcoef(a, b)[0, 1]) < 0.85

    def test_the_spreads_scale_and_the_scale_reaches_the_cases(self) -> None:
        sp = R.SPREADS.scaled(2.0)
        assert sp.cov_lo_sd == 2 * R.SPREADS.cov_lo_sd and sp.cov_rho == R.SPREADS.cov_rho
        # the spread of the perception variants reaches the C-ADAPT verdict: ten times as noisy
        # per-episode values cannot exist for a value in [0, 1] at a mean of 0.95
        case = next(c for c in R.CASES if c.id == "C-ADAPT/perfect")
        assert case.run(1.0).label == WIN and case.run(3.0).label == WIN
        assert R.label_at(case, 6.0) is None  # no data in [0, 1] have a mean of 0.95 and sd 0.42

    def test_the_break_even_is_the_largest_scale_that_keeps_the_label(self) -> None:
        case = next(c for c in R.CASES if c.id == "C-ADAPT/perfect")
        k = R.break_even(case, steps=6)
        assert (
            k is not None and 2.5 < k < 4.0
        )  # 3 times the stated spreads still win, 4 cannot exist
        assert R.label_at(case, k) == WIN and R.label_at(case, k * 1.3) != WIN
        assert R.break_even(next(c for c in R.CASES if c.id == "C-PLAN.P0/cost/perfect")) is None

    def test_the_cases_go_through_the_code_the_benchmark_uses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []

        def spy(name: str, fn: Any) -> Any:
            def wrapped(*a: Any, **k: Any) -> Any:
                seen.append(name)
                return fn(*a, **k)

            return wrapped

        for name in ("claim_records", "search_output", "paired_difference"):
            monkeypatch.setattr(R, name, spy(name, getattr(R, name)))
        monkeypatch.setattr(R.mdiag, "records", spy("mdiag.records", R.mdiag.records))
        by_id = {c.id: c for c in R.CASES}
        for cid in (
            "C-ADAPT/perfect",
            "C-SEARCH/perfect",
            "C-CAUSE/perfect",
            "C-DIAG/perfect",
            "C-PLAN.P0/safety/perfect",
        ):
            by_id[cid].run(1.0)
        assert seen == [
            "claim_records",
            "search_output",
            "paired_difference",
            "mdiag.records",
            "claim_records",
        ]


class TestFloors:
    def test_the_floors_and_factors_of_the_preregistration(self) -> None:
        by = {f.n: f for f in R.floors()}
        # section 6: 0.080 at n = 300, 0.048 at 500, 0.034 at 700; 0.216 s and 0.168 s
        assert by[300].betting == pytest.approx(0.080, abs=5e-4)
        assert by[500].betting == pytest.approx(0.048, abs=5e-4)
        assert by[700].betting == pytest.approx(0.034, abs=5e-4)
        assert by[300].spread_factor == pytest.approx(0.216, abs=5e-4)
        assert by[500].spread_factor == pytest.approx(0.168, abs=5e-4)
        assert all(f.alpha == ALPHA_PRIMARY for f in by.values())

    def test_a_smaller_level_raises_both(self) -> None:
        a = {f.n: f for f in R.floors(ALPHA_PRIMARY)}
        b = {f.n: f for f in R.floors(ALPHA_PERC)}
        assert all(
            b[n].betting > a[n].betting and b[n].spread_factor > a[n].spread_factor for n in a
        )

    def test_the_floor_falls_with_n(self) -> None:
        fs = R.floors()
        assert [f.n for f in fs] == [100, 300, 500, 700]
        assert all(x.betting > y.betting for x, y in zip(fs, fs[1:], strict=False))

    def test_the_array_form_of_the_half_width_is_the_scalar_one(self) -> None:
        sd = np.array([0.0, 0.1, 0.5])
        assert R.spread_halfwidths(sd, 300, 0.05 / 9) == pytest.approx(
            [_spread_halfwidth(300, float(s), 0.05 / 9) for s in sd], rel=1e-13
        )


class TestCoverageSimulation:
    def test_it_is_deterministic_with_a_fixed_seed(self) -> None:
        shapes = {"binary 0.3": R.COVERAGE_CASES["binary 0.3"]}
        a = R.coverage_sim(2000, seed=2, cases=shapes)
        b = R.coverage_sim(2000, seed=2, cases=shapes)
        c = R.coverage_sim(2000, seed=3, cases=shapes)
        assert a == b and a != c

    def test_a_stricter_guard_never_misses_more_and_never_uses_the_sequence_more(self) -> None:
        shapes = {k: R.COVERAGE_CASES[k] for k in ("binary 0.3", "lump 8 p.075 base 2")}
        rows = R.coverage_sim(3000, seed=2, cases=shapes)
        for name in shapes:
            by = {r.guard: r for r in rows if r.case == name}
            assert by[25].miss >= by[50].miss >= by[100].miss
            assert by[25].used_at_300 >= by[50].used_at_300 >= by[100].used_at_300
            assert 0.0 <= by[50].used_at_300 <= 1.0

    def test_the_simulation_uses_the_sample_sd_with_n_minus_one_and_the_registered_halfwidth(
        self,
    ) -> None:
        # one path repeated: 100 values on a grid (kurtosis 1.8, under the limit of 2 at n = 100).
        # A true mean 0.3 per cent inside the half-width is covered and 0.3 per cent outside it is
        # missed; with ddof = 0 the half-width is 0.5 per cent shorter and the first would miss
        n = 100
        vals = np.linspace(-1.0, 1.0, n)
        half = float(R.spread_halfwidths(float(np.std(vals, ddof=1)), n, ALPHA_PRIMARY))

        def draw(_rng: Any, shape: tuple[int, int]) -> Any:
            return np.tile(vals, (shape[0], 1))

        def miss(mu: float) -> float:
            (row,) = R.coverage_sim(
                1000, cases={"c": (draw, mu)}, guards=(50,), looks=(n,), alpha=ALPHA_PRIMARY
            )
            assert row.used_at_300 == 0.0  # no look at 300: only the miss is read here
            return row.miss

        centre = float(vals.mean())
        assert miss(centre + 0.997 * half) == 0.0
        assert miss(centre - 0.997 * half) == 0.0
        assert miss(centre + 1.003 * half) == 1.0
        assert miss(centre - 1.003 * half) == 1.0
        # and the guard is read: the same path with a limit it fails (G = 100 wants kurtosis <= 1)
        (row,) = R.coverage_sim(
            1000, cases={"c": (draw, centre + 5 * half)}, guards=(100,), looks=(n,)
        )
        assert row.miss == 0.0

    def test_the_looks_start_at_the_minimum_number_of_pairs(self) -> None:
        assert R.COVERAGE_LOOKS[0] == 100 and R.COVERAGE_LOOKS[-1] == 500
        assert R.COVERAGE_GUARDS == (25, 50, 100)
        assert len(R.COVERAGE_CASES) == 13

    def test_a_shape_the_guard_lets_through_stays_inside_its_level(self) -> None:
        # a rate of 0.3 has kurtosis 1.8: the sequence is used at every look and misses far less
        # than its level
        rows = R.coverage_sim(4000, seed=2, cases={"b": R.COVERAGE_CASES["binary 0.3"]})
        row = next(r for r in rows if r.guard == 50)
        assert row.used_at_300 == 1.0 and row.miss_over_alpha < 1.0

    def test_the_registered_guard_misses_by_several_times_alpha_on_a_rare_lump(self) -> None:
        # The measured finding of P12c: on "lump 8 p.02 base .5" the registered guard (kurtosis at
        # most n / 50) lets the sequence through at some looks (it is used at n = 300 in none of
        # the runs, but at n = 100 and 150 in some) and it misses at several times its level, 4.5
        # at 40,000 runs. The whole rule misses at most this plus alpha. The test pins the order
        # of magnitude so that a change of guard, of r or of the formula shows; G = 100 does not.
        shapes = {"lump": R.COVERAGE_CASES["lump 8 p.02 base .5"]}
        rows = R.coverage_sim(20000, seed=2, cases=shapes)
        by = {r.guard: r for r in rows}
        assert 2.0 < by[50].miss_over_alpha < 8.0
        assert by[100].miss_over_alpha < 1.0
        assert by[25].miss_over_alpha > by[50].miss_over_alpha


@pytest.fixture(scope="module")
def records() -> list[Any]:
    return R.reach_records(sim_reps=1000)


class TestRecords:
    def test_every_record_is_labelled_made_up_data(self, records: list[Any]) -> None:
        assert records and all(r.name.startswith("reach.") for r in records)
        assert all(r.measured_on.startswith("MADE-UP DATA") for r in records)
        assert all(r.extra["made_up"] == 1 for r in records)

    def test_the_records_use_none_of_the_words_the_report_forbids(self, records: list[Any]) -> None:
        text = " ".join(
            [r.measured_on for r in records] + [str(v) for r in records for v in r.extra.values()]
        )
        assert FORBIDDEN.findall(text) == []

    def test_floors_verdicts_and_simulation_are_all_there(self, records: list[Any]) -> None:
        names = [r.name for r in records]
        assert [n for n in names if n.startswith("reach.floor.")] == [
            f"reach.floor.n{n}" for n in R.FLOOR_SIZES
        ]
        assert sum(n.startswith("reach.spread_factor.") for n in names) == len(R.FLOOR_SIZES)
        verdicts = [r for r in records if r.name.startswith("reach.verdict.")]
        assert len(verdicts) == len(R.CASES)
        assert sum(n.startswith("reach.cover.") for n in names) == 13 * 3

    def test_a_verdict_record_says_whether_the_label_was_reached_and_whether_it_can_be(
        self, records: list[Any]
    ) -> None:
        by = {r.name.removeprefix("reach.verdict."): r for r in records if ".verdict." in r.name}
        win = by["C-MON/perfect"]
        assert win.value == 1.0 and win.extra["label"] == "win" and win.extra["reachable"] == 1
        cannot = by["C-PLAN.P0/cost/perfect"]
        assert cannot.value == 0.0 and cannot.extra["label"] == "inconclusive"
        assert cannot.extra["reachable"] == 0 and "kurtosis" in str(cannot.extra["why"])

    def test_the_simulation_rows_carry_a_wilson_interval_and_the_registered_guard(
        self, records: list[Any]
    ) -> None:
        row = next(r for r in records if r.name == "reach.cover.binary 0.3.G50")
        assert row.ci_kind == "wilson-0.95" and row.ci_lo is not None and row.ci_hi is not None
        assert row.ci_lo <= row.value <= row.ci_hi
        assert row.extra["registered_guard"] == 1 and row.n == 1000
        other = next(r for r in records if r.name == "reach.cover.binary 0.3.G25")
        assert other.extra["registered_guard"] == 0

    def test_quick_runs_fewer_simulated_runs_than_a_full_run(self) -> None:
        import inspect

        src = inspect.getsource(R.reach_records)
        assert "2000 if quick else 40000" in src


class TestTheStage:
    def context(self, mode: str = "quick") -> StageContext:
        return StageContext(
            mode=mode, counts=FULL, runner=Runner(workers=1, headroom=lambda: 40.0),
            context=None, seeds=(), seed_role="made-up", tuned=PROVISIONAL,  # type: ignore[arg-type]
            primary_variant="V2L", monitor_mode="opt", root=Path("."),
        )  # fmt: skip

    def test_it_is_in_the_registry_first_with_no_row_of_the_budget(self) -> None:
        assert stages.STAGE_KEYS[0] == "reach"
        stage = stages.STAGES[0]
        assert stage.rows == () and stage.needs == ()

    def test_the_stage_writes_the_records_and_says_how_many_cases_give_their_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[bool] = []
        real = R.reach_records

        def spy(quick: bool = False, sim_reps: int | None = None) -> list[Any]:
            seen.append(quick)
            return real(quick, sim_reps=500)

        monkeypatch.setattr(R, "reach_records", spy)
        out = run_pipeline(self.context(), ("reach",))
        (stage,) = out.outputs
        assert stage.status == "done" and out.complete is False  # the other stages were skipped
        assert seen == [True]
        assert any("reachability:" in n and f"of {len(R.CASES)} cases" in n for n in stage.notes)
        assert any("C-PLAN.P0/cost/perfect" in n for n in stage.notes)

    def test_a_case_that_does_not_behave_as_recorded_makes_the_stage_partial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = R.reach_records

        def spy(quick: bool = False, sim_reps: int | None = None) -> list[Any]:
            recs = real(quick, sim_reps=500)
            return [
                dataclasses.replace(r, value=0.0) if r.name == "reach.verdict.C-MON/perfect" else r
                for r in recs
            ]

        monkeypatch.setattr(R, "reach_records", spy)
        (stage,) = run_pipeline(self.context(), ("reach",)).outputs
        assert stage.status == "partial"
        assert any("DO NOT BEHAVE AS RECORDED: C-MON/perfect" in n for n in stage.notes)

    def test_a_full_run_simulates_more_than_a_quick_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[bool] = []

        def spy(quick: bool = False, sim_reps: int | None = None) -> list[Any]:
            seen.append(quick)
            return []

        monkeypatch.setattr(R, "reach_records", spy)
        run_pipeline(self.context("full"), ("reach",))
        assert seen == [False]


def test_the_floor_text_of_the_preregistration_matches_the_code() -> None:
    """The numbers prereg section 6 quotes are the ones ``floors`` computes."""
    docs = Path(__file__).resolve().parents[2] / "docs"
    # the draft is renamed to PREREGISTRATION.md when it is frozen; the text is the same file
    prereg = docs / "PREREGISTRATION.draft.md"
    text = (prereg if prereg.exists() else docs / "PREREGISTRATION.md").read_text(encoding="utf-8")
    fs = {f.n: f for f in R.floors()}
    for quoted in (
        f"{fs[300].betting:.3f} at n = 300",
        f"{fs[500].betting:.3f} at n = 500",
        f"{fs[700].betting:.3f} at n = 700",
        f"{fs[300].spread_factor:.3f} s at n = 300",
        f"{fs[500].spread_factor:.3f} s at n = 500",
    ):
        assert quoted in text, quoted
    assert math.isclose(fs[300].betting, 0.08, abs_tol=5e-4)
