"""P11c: the parts of the F2 episode that need no sibling.

No ``f2`` marker on purpose: these pass on a machine without the siblings, as the rule of 0002
section 1 requires. They cover the caching-rule arithmetic, the plant of an F2 episode, the arms,
the sim-only reads and the stage's seed rule.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from closed_loop.audit.seeds import f2_seed
from closed_loop.core.timegrid import DAY_Q, HORIZON_Q
from closed_loop.core.trace import Trace
from closed_loop.core.types import Op
from closed_loop.eval.arms import ARMS
from closed_loop.eval.context import PreparedContext, use_context
from closed_loop.f2.episode import (
    F2_ARMS,
    PASS_THROUGH_H,
    SimClock,
    _expected_runs,
    f2_params,
)
from closed_loop.plant import make_plant
from closed_loop.plant.params import ASSET_IDS, PlantParams
from closed_loop.plant.scenario import Scenario

ROOT = Path(__file__).resolve().parents[2]
RUN = list(Op).index(Op.RUN)
STOP = list(Op).index(Op.STOP)


def trace_with(stops: dict[str, list[tuple[int, int]]], n_days: int = 6) -> Trace:
    """A trace in which every asset runs except for the row ranges ``[a, b)`` given."""
    n_q = n_days * DAY_Q
    tr = Trace.empty(n_q, ASSET_IDS)
    for aid in ASSET_IDS:
        tr.op[aid][:] = RUN
        for a, b in stops.get(aid, []):
            tr.op[aid][a:b] = STOP
    return tr


class TestExpectedRuns:
    """The caching rule, counted from the plant's op rows alone."""

    def test_nothing_stops_so_each_asset_runs_the_pipeline_once(self) -> None:
        assert _expected_runs(trace_with({}), 6) == 4

    def test_a_stop_and_restart_across_an_update_costs_one_more_run(self) -> None:
        # stopped from row 100 to 200: running at updates 0, 1 and 3, so one extra for that asset
        assert _expected_runs(trace_with({"PU-201": [(100, 200)]}), 6) == 5

    def test_a_blink_between_two_updates_costs_one_more_run(self) -> None:
        assert _expected_runs(trace_with({"CR-501": [(10, 20)]}), 6) == 5

    def test_a_stop_that_never_ends_costs_nothing_more(self) -> None:
        # the crusher fails at row 300 and is never started again: no restart, no run
        assert _expected_runs(trace_with({"CR-501": [(300, 6 * DAY_Q)]}), 6) == 4

    def test_two_restarts_between_the_same_pair_of_updates_cost_one_run(self) -> None:
        assert _expected_runs(trace_with({"CV-101": [(10, 20), (40, 50)]}), 6) == 5

    def test_each_restart_in_a_different_interval_costs_its_own_run(self) -> None:
        assert _expected_runs(trace_with({"CV-101": [(10, 20), (200, 210)]}), 6) == 6

    def test_an_asset_stopped_at_an_update_is_held_and_the_next_running_update_counts(
        self,
    ) -> None:
        # stopped over the update at row 191 (day 2); the pair (day 1, day 3) holds the stop
        assert _expected_runs(trace_with({"PU-202": [(150, 250)]}), 6) == 5


class TestThePlantOfAnEpisode:
    def test_only_the_approval_is_made_a_pass_through(self) -> None:
        base = PlantParams()
        f2 = f2_params(base)
        assert f2.approver_median_h == PASS_THROUGH_H and f2.approver_catch_p == 0.0
        changed = {k for k in base.__dataclass_fields__ if getattr(base, k) != getattr(f2, k)}
        assert changed == {"approver_median_h", "approver_catch_p"}

    def test_the_exogenous_streams_are_those_of_the_f1_plant(self) -> None:
        """With nobody acting, the F1 plant and the F2 plant of one seed are the same world."""
        base = PlantParams()
        traces = []
        for params in (base, f2_params(base)):
            plant = make_plant(params)
            plant.reset(1023, Scenario())
            for _ in range(HORIZON_Q):
                plant.step([])
            traces.append(plant.trace())
        a, b = traces
        for aid in ASSET_IDS:
            for name in ("op", "true_rul_days", "health_bin", "uncontrolled_failure", "alarm"):
                assert np.array_equal(getattr(a, name)[aid], getattr(b, name)[aid]), (aid, name)
        assert np.array_equal(a.throughput, b.throughput)
        assert a.uncontrolled_failure["CV-101"].any() or a.uncontrolled_failure["CR-501"].any()

    def test_the_sim_clock_starts_at_the_plants_six_in_the_morning(self) -> None:
        clock = SimClock()
        assert (clock().hour, clock().minute, clock().tzinfo is not None) == (6, 0, True)
        clock.quarter = 5
        assert (clock().hour, clock().minute) == (7, 15)


class TestTheArms:
    def test_names_and_conditions(self) -> None:
        assert list(F2_ARMS) == ["F2-B3", "F2-P0", "F2-P1"]
        assert all(a.conditions == ("nominal",) for a in F2_ARMS.values())

    def test_only_the_p1_arm_has_a_shield_and_it_is_the_f1_one(self) -> None:
        assert [a.shield for a in F2_ARMS.values()] == [False, False, True]
        f1 = ARMS["P1"]
        f2 = F2_ARMS["F2-P1"]
        assert (f2.spec, f2.shield_set, f2.perception, f2.interlock) == (
            f1.spec,
            f1.shield_set,
            f1.perception,
            f1.interlock,
        )

    def test_the_proposer_has_no_diagnoser_because_the_diagnosis_is_asset_healths(
        self, small_context: PreparedContext
    ) -> None:
        with use_context(small_context):
            for arm in F2_ARMS.values():
                assert arm.proposer().diagnoser is None  # type: ignore[attr-defined]


class TestSimOnlyReads:
    """The controller side of the F2 loop reads no truth (the rule of 0002 section 3.5)."""

    @staticmethod
    def readers(path: Path, attribute: str) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found: set[str] = set()
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)] + [tree]:
            body = cls.body
            for node in body:
                if isinstance(node, ast.FunctionDef):
                    name = f"{getattr(cls, 'name', '')}.{node.name}".lstrip(".")
                    if any(
                        isinstance(n, ast.Attribute) and n.attr == attribute for n in ast.walk(node)
                    ):
                        found.add(name)
        return found

    def test_nothing_reads_the_plants_truth_view(self) -> None:
        for module in ("episode", "telemetry", "replay"):
            assert self.readers(ROOT / "closed_loop" / "f2" / f"{module}.py", "truth") == set()

    def test_the_plants_internals_are_read_in_three_named_places_only(self) -> None:
        episode = self.readers(ROOT / "closed_loop" / "f2" / "episode.py", "des")
        telemetry = self.readers(ROOT / "closed_loop" / "f2" / "telemetry.py", "des")
        assert episode == {"AgentChain.__init__", "AgentChain._truth_mode"}
        assert telemetry == {"AssetTelemetry._start"}


class TestTheStageSeeds:
    def test_a_quick_run_uses_development_seeds_above_the_planners_own(self) -> None:
        from types import SimpleNamespace

        from closed_loop.eval.stage_f2 import seeds_for

        sc = SimpleNamespace(prereg_sha=None, published_f2_seeds=None)
        assert seeds_for(sc, 3) == [1100, 1101, 1102]  # type: ignore[arg-type]

    def test_a_full_run_takes_the_seeds_the_guard_drew_and_derives_none_of_its_own(self) -> None:
        # 0003 A16: seeds_for no longer calls f2_seed. The context carries what the held-out guard
        # handed out (and the ledger counted); the stage takes the first n of them
        from types import SimpleNamespace

        from closed_loop.eval.stage_f2 import seeds_for

        sha = "ab" * 32
        drawn = tuple(f2_seed(sha, i) for i in range(5))
        sc = SimpleNamespace(prereg_sha=sha, published_f2_seeds=drawn)
        assert seeds_for(sc, 3) == list(drawn[:3])  # type: ignore[arg-type]
        assert seeds_for(sc, 5) == list(drawn)  # type: ignore[arg-type]

    def test_a_full_run_that_drew_none_is_an_error_not_a_quiet_derivation(self) -> None:
        from types import SimpleNamespace

        from closed_loop.eval.stage_f2 import seeds_for

        sc = SimpleNamespace(prereg_sha="ab" * 32, published_f2_seeds=None)
        with pytest.raises(ValueError, match="held-out guard"):
            seeds_for(sc, 3)  # type: ignore[arg-type]

    def test_too_few_drawn_seeds_is_an_error(self) -> None:
        from types import SimpleNamespace

        from closed_loop.eval.stage_f2 import seeds_for

        sc = SimpleNamespace(prereg_sha="ab" * 32, published_f2_seeds=(1, 2))
        with pytest.raises(ValueError, match="3 F2 seeds are needed"):
            seeds_for(sc, 3)  # type: ignore[arg-type]

    def test_seeds_for_does_not_call_the_seed_function_itself(self) -> None:
        import ast
        import inspect
        import textwrap

        from closed_loop.eval import stage_f2

        tree = ast.parse(textwrap.dedent(inspect.getsource(stage_f2.seeds_for)))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "f2_seed" not in names
        assert not hasattr(stage_f2, "f2_seed")  # the module does not even import it

    def test_the_f2_stage_is_built(self) -> None:
        from closed_loop.eval.stages import STAGES, missing_modules

        stage = next(s for s in STAGES if s.key == "f2")
        assert missing_modules(stage) == []


@pytest.mark.parametrize("module", ["telemetry", "episode", "mismatch", "replay"])
def test_the_new_modules_import_without_a_sibling_or_pandas(module: str) -> None:
    import os
    import subprocess
    import sys

    code = (
        f"import closed_loop.f2.{module}, sys\n"
        "bad = [m for m in ('asset_health', 'maintenance_agent', 'plant_copilot', 'pandas', "
        "'sklearn') if m in sys.modules]\nprint(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert out.stdout.strip().splitlines()[-1] == "[]"
