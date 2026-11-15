"""P11c: the F2 stage plugs into the benchmark runner as stages.py describes.

The records are tested on the episodes of the session fixture; the whole stage (three episodes and
the estimator mismatch) runs once, marked ``slow``, through ``stages.run_pipeline``, the way
``benchmarks/run.py`` calls it. This stage has no search of its own: the replay set is what the
search stage leaves in ``sc.shared`` (``tests/eval/test_stage_f2_replay.py`` tests that hand-off),
and with no search stage in the invocation the replay set is empty and the stage says so.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval import stage_f2
from closed_loop.eval.claims import claim_name
from closed_loop.eval.context import PreparedContext
from closed_loop.eval.counts import QUICK
from closed_loop.eval.runner import Runner
from closed_loop.eval.stages import NotBuiltError, StageContext, run_pipeline
from closed_loop.eval.tuned import PROVISIONAL
from closed_loop.f2 import F2Unavailable

from .conftest import Timed

pytestmark = pytest.mark.f2


@pytest.fixture(scope="module")
def records(f2_episodes: dict[str, Timed]) -> dict[str, dict[str, ResultRecord]]:
    out: dict[str, dict[str, ResultRecord]] = {}
    for arm, timed in f2_episodes.items():
        recs = stage_f2._arm_records(arm, [timed.record], 1, "", "1 F2 episode per arm")
        out[arm] = {r.name: r for r in recs}
    return out


class TestTheArmRecords:
    def test_the_completion_record_is_what_the_claim_reads(
        self, records: dict[str, dict[str, ResultRecord]]
    ) -> None:
        for arm, recs in records.items():
            rec = recs[claim_name("C-TOGETHER", "arm", arm)]
            assert (rec.extra["completed"], rec.extra["tried"], rec.n) == (1, 1, 1)
            assert rec.fidelity is Fidelity.F2 and rec.script == "closed_loop.eval.stage_f2"

    def test_an_arm_that_did_not_complete_carries_its_cause_and_nothing_else(self) -> None:
        recs = stage_f2._arm_records("F2-B3", [], 3, "seed 1100: KeyError: boom", "3 episodes")
        assert len(recs) == 1
        assert recs[0].extra["completed"] == 0 and "boom" in str(recs[0].extra["cause"])

    def test_the_counted_wrong_diagnosis_metric_is_a_record_with_its_denominators(
        self, records: dict[str, dict[str, ResultRecord]]
    ) -> None:
        for arm, recs in records.items():
            signed = recs[f"f2.{arm}.wrong_compliant_signed"]
            presented = recs[f"f2.{arm}.wrong_compliant_presented"]
            assert signed.extra["k"] <= presented.extra["k"]
            assert "sixteen rules" in signed.measured_on
            assert signed.extra["drafts"] == (
                signed.extra["repair_drafts"]
                + signed.extra["inspection_first_drafts"]
                + signed.extra["alerts_without_a_draft"]
            )

    def test_the_pieces_the_telemetry_dropped_are_a_record_per_arm_with_their_meaning(
        self, records: dict[str, dict[str, ResultRecord]]
    ) -> None:
        """P12g, G2: a silent skip is what this project does not do."""
        for arm, recs in records.items():
            rec = recs[f"f2.dropped_pieces.{arm}"]
            assert rec.n == 1 and rec.value == rec.extra["total"] >= 0.0
            assert 0 <= rec.extra["k"] <= rec.n
            assert "restarted past the failure day" in rec.measured_on
            assert "life's first start" in rec.measured_on and "no more rows" in rec.measured_on

    def test_the_record_counts_the_total_and_the_episodes_with_any(self) -> None:
        def episode(dropped: int) -> Any:
            return SimpleNamespace(
                result=SimpleNamespace(
                    throughput=0.9, cost=1.0, unplanned_failures=0, arm="F2-P1", seed=1,
                ),
                log=SimpleNamespace(
                    run_asset_calls=4, expected_run_asset_calls=4, repair_drafts=0,
                    inspection_drafts=0, runs_without_a_draft=0, wrong_compliant_signed=0,
                    wrong_compliant_presented=0, dropped_pieces=dropped,
                ),
            )  # fmt: skip

        recs = {
            r.name: r
            for r in stage_f2._arm_records(
                "F2-P1", [episode(0), episode(2), episode(1), episode(0)], 4, "", "4 episodes"
            )
        }
        rec = recs["f2.dropped_pieces.F2-P1"]
        assert (rec.value, rec.n, rec.extra["k"], rec.extra["total"]) == (3.0, 4, 2, 3)

    def test_the_call_count_is_recorded_against_the_rule_and_the_clock_is_not_a_record(
        self, records: dict[str, dict[str, ResultRecord]]
    ) -> None:
        # wall-clock time goes to the run log (results.json is the same bytes on every run)
        for arm, recs in records.items():
            calls = recs[f"f2.{arm}.run_asset_calls"]
            assert calls.extra["equal_to_caching_rule"] == 1
            assert f"f2.{arm}.seconds_per_episode" not in recs

    def test_every_record_is_valid_and_says_what_it_was_measured_on(
        self, records: dict[str, dict[str, ResultRecord]]
    ) -> None:
        for recs in records.values():
            for rec in recs.values():
                assert rec.measured_on and rec.script and rec.fidelity is Fidelity.F2


def test_the_stage_reports_not_built_when_a_sibling_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> None:
        raise F2Unavailable("no asset_health here")

    monkeypatch.setattr(stage_f2, "require_siblings", missing)
    sc = SimpleNamespace(counts=QUICK, prereg_sha=None, published_f2_seeds=None)
    with pytest.raises(NotBuiltError, match="sibling"):
        stage_f2.run(sc)  # type: ignore[arg-type]


@pytest.mark.slow
def test_the_whole_stage_runs_through_the_pipeline(small_context: PreparedContext) -> None:
    counts = dataclasses.replace(QUICK, f2_episodes=1, f2_replays=1, search_budget=20)
    runner = Runner(workers=1, headroom=lambda: 40.0, timeout_s=None)
    sc = StageContext(
        mode="quick",
        counts=counts,
        runner=runner,
        context=small_context,
        seeds=(1100,),
        seed_role="dev",
        tuned=PROVISIONAL,
        primary_variant="V3",
        monitor_mode="proxy",
        root=Path("."),
    )
    result = run_pipeline(sc, ("f2",))
    (out,) = result.outputs
    assert out.key == "f2" and out.status == "done" and "f2" not in result.not_built
    by_name = {r.name: r for r in out.records}
    for arm in ("F2-B3", "F2-P0", "F2-P1"):
        rec = by_name[claim_name("C-TOGETHER", "arm", arm)]
        assert (rec.extra["completed"], rec.extra["tried"]) == (1, 1), rec.extra
    confirmed = by_name[claim_name("C-TOGETHER", "confirmed")]
    assert confirmed.n == confirmed.extra["requested"] or confirmed.n < confirmed.extra["requested"]
    reasons = Counter({n: r.value for n, r in by_name.items() if n.startswith("f2.replay.reason.")})
    assert sum(reasons.values()) == confirmed.n - confirmed.extra["k"]
    mismatch = by_name["f2.estimator_mismatch.mae_diff_days"]
    assert mismatch.n == 20 and mismatch.fidelity is Fidelity.F2
    # what C-TOGETHER reads is there
    kinds: dict[str, Any] = {r.name: r.extra for r in out.records}
    assert claim_name("C-TOGETHER", "confirmed") in kinds
