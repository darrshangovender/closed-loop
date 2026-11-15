"""``eval.counts``: the one place the counts, targets and registered levels live."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from closed_loop.eval.counts import (
    ALPHA_DESCRIPTIVE,
    ALPHA_PERC,
    ALPHA_PRIMARY,
    FULL,
    N_PRIMARY_CLAIMS,
    QUICK,
    QUICK_FIRST_DEV_SEED,
    Counts,
    stop_rule_for,
)


class TestRegisteredCounts:
    def test_the_full_counts_are_the_preregistered_maxima(self) -> None:
        # prereg section 3 after 0003 A14: 500 per cell for rule arms, perception variants and
        # hostile arms, 300 for planner arms, batches of 50, 600 causal lives (300 train, 300
        # test), 60 F2 episodes and up to 25 replays, 10 runs of 200 evaluations
        assert (FULL.batch, FULL.n_max_perception, FULL.n_max_policy, FULL.n_max_hostile) == (
            50, 500, 500, 500,
        )  # fmt: skip
        assert FULL.n_max_planner == 300
        assert FULL.causal_train_lives == 300 and FULL.mdiag_train_seeds == 300
        assert FULL.conformance_episodes == 200 and FULL.exact_dp_beliefs == 50
        assert (FULL.search_runs, FULL.search_budget) == (10, 200)
        assert (FULL.f2_episodes, FULL.f2_replays) == (60, 25)

    def test_the_levels(self) -> None:
        assert N_PRIMARY_CLAIMS == 9
        assert ALPHA_PRIMARY == pytest.approx(0.05 / 9)
        assert ALPHA_PERC == pytest.approx(0.05 / 18) and ALPHA_DESCRIPTIVE == 0.05

    def test_a_group_has_a_maximum(self) -> None:
        assert [FULL.n_max(g) for g in ("perception", "policy", "hostile")] == [500, 500, 500]
        assert FULL.n_max("planner") == 300
        with pytest.raises(ValueError, match="unknown cell group"):
            FULL.n_max("elsewhere")

    def test_the_run_needs_as_many_seeds_as_its_largest_cell(self) -> None:
        assert FULL.largest_cell == 700  # the C-MON cell: P1 unshifted
        assert dataclasses.replace(FULL, mon_nominal=400).largest_cell == 500
        assert dataclasses.replace(FULL, n_max_policy=900).largest_cell == 900

    def test_the_monitor_cells_have_their_own_fixed_numbers(self) -> None:
        # 0003 A15: P1 at 700 unshifted and 300 shifted; every other planner arm keeps 300
        assert (FULL.mon_nominal, FULL.mon_shifted) == (700, 300) and FULL.n_max_planner == 300
        assert FULL.cell_limit("P1", "nominal", "planner") == 700
        assert FULL.cell_limit("P1", "shifted", "planner") == 300
        assert FULL.cell_limit("P0", "nominal", "planner") == 300
        assert FULL.cell_limit("P1-I", "nominal", "planner") == 300
        assert FULL.cell_limit("B3-S", "nominal", "policy") == 500
        assert FULL.cell_limit("H-C1", "nominal", "hostile") == 500

    def test_the_single_pre_freeze_adjustment_is_a_one_file_change(self) -> None:
        # a cut of n_max is a `replace` of this dataclass and reaches every cell of the group
        cut = dataclasses.replace(FULL, n_max_policy=350)
        assert cut.n_max("policy") == 350 and cut.n_max("hostile") == 500
        assert cut.n_max("planner") == 300  # the planner arms have a limit of their own


class TestQuick:
    def test_every_count_is_divided_by_25_with_a_floor_of_4(self) -> None:
        # 500 / 25 = 20 per cell, then the cap of QUICK_CELL_CAP on the F1 cells
        assert QUICK.n_max_perception == QUICK.n_max_policy == QUICK.n_max_hostile == 12
        assert QUICK.open_loop_seeds == 12 and QUICK.conformance_episodes == 8
        assert QUICK.batch == 4  # 50 / 25 = 2, raised to the floor
        assert QUICK.exact_dp_beliefs == 4  # 50 / 25 = 2, raised to the floor
        assert QUICK.mdiag_train_seeds == 12 and QUICK.mdiag_test_seeds == 12
        assert QUICK.causal_test_seeds == 4  # 75 / 25 = 3, raised to the floor

    def test_the_cell_cap_is_the_cut_taken_for_the_eight_minutes(self) -> None:
        from closed_loop.eval.counts import QUICK_CELL_CAP

        assert QUICK_CELL_CAP == 12 and QUICK_CELL_CAP < FULL.n_max_policy // 25
        assert QUICK.largest_cell == 12 and QUICK.conformance_episodes == 8  # not capped
        assert QUICK.batch == 4 and QUICK_CELL_CAP % QUICK.batch == 0  # whole batches

    def test_the_counts_0001_states_outright(self) -> None:
        assert (QUICK.f2_episodes, QUICK.f2_replays) == (3, 3)

    def test_the_search_is_cut_below_what_0001_states_to_fit_the_eight_minutes(self) -> None:
        from closed_loop.eval.counts import QUICK_SEARCH_BUDGET, QUICK_SEARCH_RUNS

        assert (QUICK.search_runs, QUICK.search_budget) == (QUICK_SEARCH_RUNS, QUICK_SEARCH_BUDGET)
        assert (QUICK_SEARCH_RUNS, QUICK_SEARCH_BUDGET) == (2, 15)
        assert (FULL.search_runs, FULL.search_budget) == (10, 200)  # the full run is untouched
        # 4 cells x runs x evaluations at 3.5 s, against 8 minutes of two workers
        assert 4 * QUICK.search_runs * QUICK.search_budget * 3.5 < 8 * 60 * 2

    def test_the_effect_models_keep_enough_lives_to_fit(self) -> None:
        # the correlational model needs 9 training lives that alerted; 12 cannot give them
        assert QUICK.causal_train_lives == 40

    def test_f0_is_cut_and_not_scaled(self) -> None:
        assert QUICK.f0_widths == (4,) and not QUICK.f0_sabotage and not QUICK.f0_stale
        assert FULL.f0_widths == (1, 2, 3, 4) and FULL.f0_sabotage and FULL.f0_stale

    def test_no_quick_count_is_larger_than_the_full_one(self) -> None:
        for f in dataclasses.fields(Counts):
            quick, full = getattr(QUICK, f.name), getattr(FULL, f.name)
            if isinstance(full, int) and not isinstance(full, bool):
                assert quick <= full or f.name == "causal_train_lives", f.name

    def test_flags_are_not_scaled_as_numbers(self) -> None:
        assert isinstance(QUICK.f0_sabotage, bool) and QUICK.f0_sabotage is False
        assert FULL.quick().episode_timeout_s == FULL.episode_timeout_s

    def test_the_quick_seeds_start_above_the_planners_fitting_seeds(self) -> None:
        assert QUICK_FIRST_DEV_SEED >= 1100  # the world model is fitted on 1000 to 1099


class TestStopRules:
    @pytest.mark.parametrize(
        ("arm", "metric", "zero_runs_on"),
        [
            ("B3-V1", "m_cov", False),
            ("B3-V7L", "m_cov", False),
            ("B3-V4-audit", "m_cov", False),
            ("H-C1", "m_viol_p13", True),
            ("H-G0", "m_viol_p13", True),
            ("S-stale", "m_viol_p13", True),
            ("B0", "m_cost", False),
            ("B3-S", "m_cost", False),
            ("P0", "m_cost", False),
            ("P2", "m_cost", False),
            ("P0-mis10", "m_cost", False),
        ],
    )
    def test_the_primary_metric_of_each_kind_of_cell(
        self, arm: str, metric: str, zero_runs_on: bool
    ) -> None:
        rule = stop_rule_for(arm)
        # 0003 A15: 0.01 for M-COV and M-COST, 0.05 for the proportions
        target = 0.05 if metric == "m_viol_p13" else 0.01
        assert (rule.metric, rule.target, rule.zero_event_runs_on) == (metric, target, zero_runs_on)

    def test_the_monitor_arm_never_stops_early(self) -> None:
        from closed_loop.eval.counts import MON_ARM

        assert MON_ARM == "P1"
        assert stop_rule_for("P1").fixed
        assert not any(
            stop_rule_for(a).fixed for a in ("P0", "P1-I", "P2", "B3-S", "H-C1", "B3-V3")
        )

    def test_every_registered_arm_has_a_rule(self) -> None:
        from closed_loop.eval.arms import ARMS

        for name in ARMS:
            assert stop_rule_for(name).metric in ("m_cov", "m_cost", "m_viol_p13")


def test_no_module_other_than_counts_fixes_a_cell_maximum() -> None:
    """The adjustment of n_max is one file: no stage hard-codes a registered maximum."""
    root = Path(__file__).resolve().parents[2] / "closed_loop" / "eval"
    for name in ("stages.py", "claims.py", "report.py"):
        text = (root / name).read_text(encoding="utf-8")
        for needle in ("n_max=500", "n_max = 500", "= 500\n"):
            assert needle not in text, (name, needle)


class TestGroupsOfArms:
    """Rule arms, perception variants and hostile arms keep 500; planner arms have 300 (A14)."""

    def test_each_arm_gets_the_maximum_of_its_kind(self) -> None:
        from closed_loop.eval.arms import ARMS
        from closed_loop.eval.stages import _group

        expected = {
            "B0": 500, "B1": 500, "B2": 500, "B3-S": 500,  # the rule arms, B3-S among them
            "P0": 300, "P1": 300, "P2": 300, "P1-corr": 300, "P1-oracle": 300, "P1-I": 300,
            "P1-step": 300, "P1-nopair": 300, "P0-noinspect": 300, "P0-mis10": 300, "P0-mis25": 300,
            "H-C0": 500, "H-C1": 500, "H-G1": 500, "S-stale": 500, "S-clause": 500,
            "B3-V1": 500, "B3-V3": 500, "B3-V7L": 500, "B3-V4-audit": 500,
        }  # fmt: skip
        for name, limit in expected.items():
            assert FULL.n_max(_group(name)) == limit, name
        assert set(expected) <= set(ARMS)
        planner = {n for n in ARMS if n.startswith("P")}
        assert planner == {n for n, v in expected.items() if v == 300}  # none left out

    def test_every_registered_arm_has_a_group_with_a_maximum(self) -> None:
        from closed_loop.eval.arms import ARMS
        from closed_loop.eval.stages import _group

        assert {FULL.n_max(_group(n)) for n in ARMS} == {300, 500}

    def test_the_planner_cap_is_cut_in_a_quick_run_like_the_other_cells(self) -> None:
        assert QUICK.n_max_planner == QUICK.n_max_policy == 12


def test_the_search_budget_is_one_constant_and_no_module_repeats_it() -> None:
    """The stage, the censoring of M-FIND and the scale of the evaluations difference of C-SEARCH
    read ``Counts.search_budget``: the figure the pre-registration had before 0003 A14 (300) is
    written in none of the modules that run or judge the search, and neither is the new one."""
    import re

    root = Path(__file__).resolve().parents[2] / "closed_loop" / "eval"
    for name in ("stages.py", "claims.py", "report.py", "runner.py"):
        text = (root / name).read_text(encoding="utf-8")
        found = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"(300|200)", line) and "budget" in line.lower()
        ]
        assert not found, (name, found)
        assert not re.search(r"300", text), name
