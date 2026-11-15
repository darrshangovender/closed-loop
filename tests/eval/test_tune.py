"""``scripts/tune.py``: the two selection rules, the dev-seed guard, and the file it writes.

The measuring is not run here (a tuning is minutes of simulation); ``tune`` is exercised with the
runner and the measurements replaced, to show that the order, the rules and the output file are
what the script says."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from closed_loop.eval.tuned import TunedError, load_tuned
from closed_loop.perception.labels import Checkpoint

TUNE_PY = Path(__file__).resolve().parents[2] / "scripts" / "tune.py"


@pytest.fixture(scope="module")
def tune_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("closed_loop_scripts_tune", TUNE_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def grid(tune: ModuleType, costs: list[float], values: list[float] | None = None) -> list[Any]:
    values = values or [float(i) for i in range(len(costs))]
    return [
        tune.GridRow(v, c, 40, c - 0.1, c + 0.1, 0.1, 3.0)
        for v, c in zip(values, costs, strict=True)
    ]


class TestChooseMinimum:
    def test_the_lowest_mean_cost_wins(self, tune_module: ModuleType) -> None:
        rows = grid(tune_module, [0.30, 0.12, 0.20], [3.0, 5.0, 7.0])
        assert tune_module.choose_minimum(rows).value == 5.0

    def test_a_tie_goes_to_the_first_value_of_the_grid(self, tune_module: ModuleType) -> None:
        rows = grid(tune_module, [0.20, 0.10, 0.10, 0.30], [3.0, 5.0, 7.0, 10.0])
        assert tune_module.choose_minimum(rows).value == 5.0

    def test_one_row(self, tune_module: ModuleType) -> None:
        assert tune_module.choose_minimum(grid(tune_module, [0.5])).value == 0.0

    def test_an_empty_grid_is_an_error(self, tune_module: ModuleType) -> None:
        with pytest.raises(ValueError, match="empty"):
            tune_module.choose_minimum([])


class TestChooseVariant:
    def row(self, tune: ModuleType, v: str, cov: float, width: float) -> Any:
        return tune.VariantRow(v, cov, width, 40)

    def test_the_highest_lower_bound_coverage_wins(self, tune_module: ModuleType) -> None:
        rows = [
            self.row(tune_module, "V2", 0.80, 1.5),
            self.row(tune_module, "V5", 0.93, 3.0),  # wider, but covers more: the rule is coverage
            self.row(tune_module, "V3", 0.88, 1.1),
        ]
        assert tune_module.choose_variant(rows).variant == "V5"

    def test_an_exact_tie_goes_to_the_narrower_interval(self, tune_module: ModuleType) -> None:
        rows = [
            self.row(tune_module, "V2", 0.9, 2.5),
            self.row(tune_module, "V4", 0.9, 1.8),
            self.row(tune_module, "V3", 0.9, 2.0),
        ]
        assert tune_module.choose_variant(rows).variant == "V4"

    def test_a_near_tie_is_not_a_tie(self, tune_module: ModuleType) -> None:
        # no tolerance is registered, so 0.9001 beats 0.9000 whatever the widths
        rows = [self.row(tune_module, "V2", 0.9000, 1.0), self.row(tune_module, "V3", 0.9001, 3.9)]
        assert tune_module.choose_variant(rows).variant == "V3"

    def test_v6_and_the_oracle_variants_are_not_candidates(self, tune_module: ModuleType) -> None:
        assert tune_module.CANDIDATES == ("V2", "V3", "V4", "V5", "V2L", "V3L")

    def test_no_candidate_is_an_error(self, tune_module: ModuleType) -> None:
        with pytest.raises(ValueError, match="no candidate"):
            tune_module.choose_variant([])


class TestChooseMode:
    """Prereg section 3: the highest detection minus false alarm; ties to the lower false alarm."""

    def row(self, tune: ModuleType, mode: str, fa: int, det: int, n: int = 40) -> Any:
        return tune.ModeRow(mode, fa, n, det, n)

    def test_the_highest_detection_minus_false_alarm_wins(self, tune_module: ModuleType) -> None:
        rows = [
            self.row(tune_module, "opt", 0, 4),  # 0.10
            self.row(tune_module, "pess", 20, 30),  # 0.25
            self.row(tune_module, "inf", 2, 24),  # 0.55
            self.row(tune_module, "proxy", 4, 16),  # 0.30
        ]
        assert tune_module.choose_mode(rows).mode == "inf"

    def test_a_tie_goes_to_the_lower_false_alarm_rate(self, tune_module: ModuleType) -> None:
        # opt: 0.70 - 0.20 = 0.50; proxy: 0.50 - 0.00 = 0.50; the same score, exactly
        rows = [self.row(tune_module, "opt", 8, 28), self.row(tune_module, "proxy", 0, 20)]
        assert rows[0].score == rows[1].score
        assert tune_module.choose_mode(rows).mode == "proxy"

    def test_a_tie_in_both_goes_to_the_order_of_the_modes(self, tune_module: ModuleType) -> None:
        rows = [self.row(tune_module, "proxy", 2, 20), self.row(tune_module, "opt", 2, 20)]
        assert tune_module.choose_mode(rows).mode == "opt"

    def test_a_near_tie_is_not_a_tie(self, tune_module: ModuleType) -> None:
        rows = [self.row(tune_module, "opt", 0, 20), self.row(tune_module, "proxy", 4, 25)]
        # 0.50 against 0.525: the second, although its false alarms are higher
        assert tune_module.choose_mode(rows).mode == "proxy"

    def test_the_rates_have_their_own_denominators(self, tune_module: ModuleType) -> None:
        r = tune_module.ModeRow("opt", 5, 50, 6, 20)
        assert (r.false_alarm, r.detection) == (
            tune_module.Fraction(1, 10),
            tune_module.Fraction(3, 10),
        )

    def test_no_mode_is_an_error(self, tune_module: ModuleType) -> None:
        with pytest.raises(ValueError, match="no monitor mode"):
            tune_module.choose_mode([])

    def test_the_modes_are_the_four_registered_ones(self, tune_module: ModuleType) -> None:
        from closed_loop.perception.monitor import MODES

        assert tuple(tune_module.MODES) == tuple(MODES)


class TestDevOnly:
    def test_dev_seeds_pass(self, tune_module: ModuleType) -> None:
        tune_module.check_dev(range(1100, 1140))

    @pytest.mark.parametrize("seed", [999, 2000, 2999, 3001, 2**63 + 5])
    def test_every_other_role_is_refused(self, tune_module: ModuleType, seed: int) -> None:
        with pytest.raises(ValueError, match="dev seeds"):
            tune_module.check_dev([1100, seed])

    def test_the_default_start_is_above_the_planners_fitting_seeds(
        self, tune_module: ModuleType
    ) -> None:
        assert tune_module.FIRST_SEED >= 1100


class TestMeanWidth:
    def cp(self, lo: float, hi: float, day: int = 30, status: str = "estimated") -> Checkpoint:
        return Checkpoint(0, "PU-201", day, lo, (lo + hi) / 2, hi, 1.0, 0.9, status)

    def test_the_width_is_the_number_of_health_bins_the_interval_touches(
        self, tune_module: ModuleType
    ) -> None:
        # edges 30, 10, 3 days: [40, 60] is bin 0 only; [5, 20] touches bins 1 and 2; [0, 50] all
        results = [
            type("R", (), {"checkpoints": (self.cp(40, 60), self.cp(5, 20), self.cp(0, 50))})()
        ]
        assert tune_module.mean_width_bins(results) == pytest.approx((1 + 2 + 4) / 3)

    def test_blind_window_checkpoints_are_left_out(self, tune_module: ModuleType) -> None:
        blind = self.cp(0, 50, day=5, status="prior")  # day 5 is inside the 23-day blind window
        results = [type("R", (), {"checkpoints": (blind, self.cp(40, 60))})()]
        assert tune_module.mean_width_bins(results) == 1.0


class TestFiles:
    def test_it_will_not_overwrite_without_force(
        self, tune_module: ModuleType, tmp_path: Path
    ) -> None:
        out = tmp_path / "tuned.json"
        tune_module.write_tuned(out, {"a": 1}, force=False)
        with pytest.raises(FileExistsError, match="--force"):
            tune_module.write_tuned(out, {"a": 2}, force=False)
        tune_module.write_tuned(out, {"a": 2}, force=True)
        assert json.loads(out.read_text(encoding="utf-8")) == {"a": 2}

    def test_the_default_output_is_the_one_the_arms_read_and_it_is_not_written_by_the_tests(
        self, tune_module: ModuleType
    ) -> None:
        from closed_loop.eval.tuned import TUNED_PATH

        before = TUNED_PATH.read_bytes() if TUNED_PATH.exists() else None
        args = tune_module.parse([])
        assert args.out == TUNED_PATH == Path(TUNED_PATH)
        # the owner ran the final tuning: parsing the arguments must leave the file as it was
        after = TUNED_PATH.read_bytes() if TUNED_PATH.exists() else None
        assert after == before

    def test_the_tables_are_printed_with_the_chosen_row_marked(
        self, tune_module: ModuleType
    ) -> None:
        rows = grid(tune_module, [0.3, 0.1], [3.0, 5.0])
        best = tune_module.choose_minimum(rows)
        text = "\n".join(tune_module.fmt_grid("B3 k", "k_b3", rows, best))
        assert "<- chosen" in text and text.count("<- chosen") == 1
        assert text.splitlines()[3].endswith("<- chosen")  # the second grid row


class TestTuneEndToEnd:
    """``tune`` with the measuring replaced: order of the steps, the rules, the file."""

    def run(
        self, tune: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> dict[str, Any]:
        import contextlib

        import closed_loop.eval.context as context_module
        import closed_loop.eval.runner as runner_module

        order: list[str] = []
        monkeypatch.setattr(context_module, "prepare_context", lambda *a, **k: object())

        class DummyRunner:
            def __init__(self, **kw: Any) -> None:
                self.kw = kw

            def __enter__(self) -> DummyRunner:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        monkeypatch.setattr(runner_module, "Runner", DummyRunner)

        def fake_variants(
            runner: Any, tuned: Any, seeds: Any, candidates: Any, log: Any
        ) -> list[Any]:
            order.append("variant")
            return [
                tune.VariantRow(v, 0.80 + 0.05 * i, 2.0, len(seeds))
                for i, v in enumerate(candidates)
            ]

        costs = {
            "k_b3": [0.3, 0.1, 0.2],
            "b2_limit": [0.2, 0.3, 0.1],
            "b1_fraction": [0.1, 0.1, 0.4],
            "delta_plan": [0.5, 0.4, 0.3],
        }
        seen: dict[str, Any] = {}

        def fake_grid(
            runner: Any, arm: Any, field: str, grid: Any, tuned: Any, seeds: Any, log: Any
        ) -> list[Any]:
            order.append(field)
            seen[field] = (arm.name, arm.perception, tuned.primary_variant, tuned.k_b3)
            return [tune.GridRow(v, c, len(seeds), c - 0.1, c + 0.1, 0.0, 2.0)
                    for v, c in zip(grid, costs[field], strict=True)]  # fmt: skip

        def fake_modes(runner: Any, tuned: Any, seeds: Any, log: Any) -> list[Any]:
            order.append("monitor_mode")
            seen["monitor_mode"] = (tuned.primary_variant, tuned.delta_plan)
            return [
                tune.ModeRow("opt", 0, 40, 4, 40),
                tune.ModeRow("pess", 20, 40, 30, 40),
                tune.ModeRow("inf", 2, 40, 24, 40),
                tune.ModeRow("proxy", 4, 40, 16, 40),
            ]

        monkeypatch.setattr(tune, "select_variant", fake_variants)
        monkeypatch.setattr(tune, "grid_search", fake_grid)
        monkeypatch.setattr(tune, "select_mode", fake_modes)
        args = argparse.Namespace(
            episodes=5, first_seed=1100, workers=2, out=tmp_path / "tuned.json", force=False,
            cache=tmp_path / "cache", lock=None, limit=88.0, candidates="V2,V3,V5",
            k_grid="3,6,9", b2_grid="4,5,6", b1_grid="0.3,0.5,0.9", delta_grid="0.01,0.02,0.05",
        )  # fmt: skip
        with contextlib.redirect_stdout(open(tmp_path / "log.txt", "w", encoding="utf-8")):
            doc = tune.tune(args, log=lambda m: None)
        self.order, self.seen, self.args = order, seen, args
        return doc

    def test_the_variant_comes_first_and_each_choice_feeds_the_next(
        self, tune_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        doc = self.run(tune_module, monkeypatch, tmp_path)
        assert self.order == [
            "variant", "k_b3", "b2_limit", "b1_fraction", "delta_plan", "monitor_mode",
        ]  # fmt: skip
        assert doc["primary_variant"] == "V5"  # the highest coverage among V2, V3, V5
        # the mode is measured last, on P1 with the chosen variant and the chosen delta_plan
        assert doc["monitor_mode"] == "inf" and self.seen["monitor_mode"] == ("V5", 0.05)
        # B3 under the chosen variant; B2 and B1 after k was chosen; P0 under the variant last
        assert self.seen["k_b3"][:3] == ("B3-V5", "V5", "V5")
        assert self.seen["b2_limit"][3] == 6.0  # k_b3 = 6.0 (the middle, mean cost 0.1) was kept
        assert self.seen["delta_plan"][:2] == ("P0", "V5")
        assert (doc["k_b3"], doc["b2_limit"], doc["b1_fraction"], doc["delta_plan"]) == (
            6.0,
            6.0,
            0.3,
            0.05,
        )  # fmt: skip (the minimum of each: the tie in B1 goes to the first value)

    def test_the_file_it_writes_is_one_the_arms_can_read_and_says_where_it_came_from(
        self, tune_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self.run(tune_module, monkeypatch, tmp_path)
        values = load_tuned(tmp_path / "tuned.json")
        assert (
            values.provisional is False and values.variant_tuned and values.primary_variant == "V5"
        )
        assert values.mode_tuned and values.monitor_mode == "inf"
        assert values.script == "scripts/tune.py" and "dev seeds 1100-1104" in values.tuned_on
        assert "planner N = 256" in values.tuned_on and "candidates V2,V3,V5" in values.tuned_on

    def test_the_tables_are_saved_next_to_the_output(
        self, tune_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self.run(tune_module, monkeypatch, tmp_path)
        text = (tmp_path / "tuned.tables.txt").read_text(encoding="utf-8")
        assert "Primary perception variant" in text and "Planner delta_plan" in text
        assert "Monitor mode (P1, dev)" in text
        assert text.count("<- chosen") == 6

    def test_an_existing_output_stops_it_before_any_measuring(
        self, tune_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import closed_loop.eval.context as context_module

        def refuse(*a: Any, **k: Any) -> Any:
            raise AssertionError("the context was fitted although the output exists")

        monkeypatch.setattr(context_module, "prepare_context", refuse)
        out = tmp_path / "tuned.json"
        out.write_text("{}", encoding="utf-8")
        args = tune_module.parse(["--out", str(out)])
        with pytest.raises(FileExistsError):
            tune_module.tune(args, log=lambda m: None)
        assert out.read_text(encoding="utf-8") == "{}"

    def test_an_unknown_candidate_is_refused(
        self, tune_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        args = tune_module.parse(["--candidates", "V6", "--out", str(tmp_path / "t.json")])
        with pytest.raises(ValueError, match="candidates must be among"):
            tune_module.tune(args, log=lambda m: None)

    def test_held_out_seeds_cannot_be_asked_for(
        self, tune_module: ModuleType, tmp_path: Path
    ) -> None:
        args = tune_module.parse(["--first-seed", "2990", "--out", str(tmp_path / "t.json")])
        with pytest.raises(ValueError, match="dev seeds"):
            tune_module.tune(args, log=lambda m: None)


def test_a_malformed_tuned_file_is_refused_by_the_loader(tmp_path: Path) -> None:
    bad = tmp_path / "tuned.json"
    bad.write_text(json.dumps({"k_b3": 7}), encoding="utf-8")
    with pytest.raises(TunedError):
        load_tuned(bad)
