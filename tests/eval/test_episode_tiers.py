"""The tiers of the episode loop: the controller sees observations, truth is read in named places.

Since 0003 A18 the loop tells the estimator what the maintenance system sees, a completed work
order (``view.jobs_done``), and nothing else about the plant's components. These tests pin that
nothing in the module reads ``plant.truth`` except the oracle's own lookup and the oracle variants'
truth, and that the life the estimator is given is the observed count.
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path
from typing import Any

import pytest

from closed_loop.eval import episode as episode_module
from closed_loop.eval.arms import ARMS
from closed_loop.eval.context import PreparedContext
from closed_loop.eval.episode import scenario_for, simulate_episode
from closed_loop.perception.estimator import TheilSenEstimator
from closed_loop.plant.params import ASSET_IDS
from closed_loop.plant.plant import F1Plant

SOURCE = Path(episode_module.__file__)
SEED = 1100
# Functions of episode.py that may read the plant's truth, and why:
ALLOWED_TRUTH_READERS = {
    "_oracle_truth": "the oracle interval variants V7 and V7L are handed the truth by definition",
    "_OracleLookup.__call__": "the oracle diagnoser of arm P1-oracle returns the truth",
    "_OracleLookup.add": "the oracle reads the sim-only cause of each alert",
}


def truth_readers() -> dict[str, int]:
    """Qualified names of the functions of episode.py that touch ``.truth`` or ``.alert_truth``."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    found: dict[str, int] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def _scope(self, node: ast.AST, name: str) -> None:
            self.stack.append(name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._scope(node, node.name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._scope(node, node.name)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr in ("truth", "alert_truth"):
                key = ".".join(self.stack) or "<module>"
                found[key] = found.get(key, 0) + 1
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


class TestWhereTruthIsRead:
    def test_only_the_oracle_touches_the_plants_truth(self) -> None:
        readers = truth_readers()
        assert readers, "the AST scan found nothing: the scan itself is broken"
        assert set(readers) <= set(ALLOWED_TRUTH_READERS), set(readers) - set(ALLOWED_TRUTH_READERS)
        assert "_oracle_truth" in readers

    def test_the_function_that_read_the_plants_life_ids_is_gone(self) -> None:
        assert not hasattr(episode_module, "maintenance_system_life_ids")
        assert "maintenance_system_life_ids" not in SOURCE.read_text(encoding="utf-8")

    def test_nothing_in_the_controller_side_wrappers_reads_truth(self) -> None:
        for name in (
            "_Counting",
            "_AuditHold",
            "_RecordingTable",
            "_OperatingClock",
            "_Perception",
        ):
            assert not [r for r in truth_readers() if r.split(".")[0] == name], name

    @pytest.fixture
    def spy(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []
        original = F1Plant.truth.fget  # type: ignore[attr-defined]

        def watched(self: F1Plant) -> Any:
            frame = sys._getframe(1)
            calls.append(frame.f_code.co_qualname)
            return original(self)

        monkeypatch.setattr(F1Plant, "truth", property(watched))
        return calls

    def test_an_arm_without_perception_never_reads_truth(
        self, small_context: PreparedContext, spy: list[str]
    ) -> None:
        simulate_episode(ARMS["B0"], SEED, scenario_for("nominal"), context=small_context)
        assert spy == []

    def test_a_perception_arm_never_reads_truth(
        self, small_context: PreparedContext, spy: list[str]
    ) -> None:
        # before A18 it read the plant's life ids once a day, 90 times
        simulate_episode(ARMS["B3-V3"], SEED, scenario_for("nominal"), context=small_context)
        assert spy == []

    def test_the_oracle_arms_read_it_only_where_the_oracle_does(
        self, small_context: PreparedContext, spy: list[str]
    ) -> None:
        simulate_episode(ARMS["B3-V7"], SEED, scenario_for("nominal"), context=small_context)
        assert set(spy) == {"_oracle_truth"}
        spy.clear()
        simulate_episode(ARMS["P1-oracle"], SEED, scenario_for("nominal"), context=small_context)
        assert set(spy) <= {"_OracleLookup.__call__"}


class TestLifeIds:
    def test_the_estimator_is_told_the_observed_count_of_completed_jobs(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        updates: list[tuple[str, int, int, int]] = []
        real_daily, real_renewed = (
            episode_module._Perception.daily,
            episode_module._Perception.renewed,
        )
        real_update = TheilSenEstimator.update

        def daily(self: Any, obs: Any) -> None:
            seen["obs"] = obs
            real_daily(self, obs)

        def renewed(self: Any, obs: Any) -> None:
            seen["obs"] = obs
            real_renewed(self, obs)

        def spying(
            self: TheilSenEstimator, asset_id: str, day: int, hi: float | None, life: int
        ) -> Any:
            told = seen["obs"].assets[asset_id].jobs_done
            updates.append((asset_id, day, life, told))
            return real_update(self, asset_id, day, hi, life)

        monkeypatch.setattr(episode_module._Perception, "daily", daily)
        monkeypatch.setattr(episode_module._Perception, "renewed", renewed)
        monkeypatch.setattr(TheilSenEstimator, "update", spying)
        arm = dataclasses.replace(ARMS["B1"], perception="V1")
        simulate_episode(arm, 1103, scenario_for("nominal"), context=small_context)
        assert updates and all(life == told for _a, _d, life, told in updates)
        assert any(life > 0 for _a, _d, life, _t in updates)  # a job was completed and seen

    def test_a_completed_job_starts_a_new_life_even_when_it_was_the_wrong_mode(
        self, small_context: PreparedContext
    ) -> None:
        """B1 on seed 1103 repairs the crusher with the wrong mode (it fits nothing, 0003 A3). The
        controller counts a completed work order, and since A18 so does perception: the lives of
        the crusher are numbered by completed jobs, whatever the plant did to the part."""
        arm = dataclasses.replace(ARMS["B1"], perception="V1")
        rec = simulate_episode(arm, 1103, scenario_for("nominal"), context=small_context)
        assert rec.result.planned_jobs >= 1 and rec.plant.stats().wrong_mode_jobs >= 1
        crusher = sorted({cp.life_id for cp in rec.result.checkpoints if cp.asset_id == "CR-501"})
        assert len(crusher) >= 2 and crusher == list(range(len(crusher)))
        lives = {
            a: {cp.life_id for cp in rec.result.checkpoints if cp.asset_id == a} for a in ASSET_IDS
        }
        assert any(len(v) > 1 for v in lives.values())
