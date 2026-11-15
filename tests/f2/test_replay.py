"""P11c: replay of a scenario vector found at F1 on the F2 stack.

The reason codes are an ordered mechanical rule, tested directly; one real replay (an F1 episode and
an F2 episode of the same arm on the same plant seed) checks that the pieces fit together.
"""

from __future__ import annotations

import numpy as np
import pytest

from closed_loop.eval.arms import ARMS
from closed_loop.eval.context import PreparedContext, use_context
from closed_loop.eval.episode import simulate_episode
from closed_loop.f2.replay import (
    JOB_PROPERTIES,
    PERCEPTION_PROPERTIES,
    REASON_CODES,
    ReplayOutcome,
    classify_unconfirmed,
    replay_theta,
)
from closed_loop.plant.scenario import Scenario
from closed_loop.search.shrink import nominal_theta

pytestmark = pytest.mark.f2


class TestTheReasonCodes:
    def test_the_four_codes_of_the_architecture_and_no_other(self) -> None:
        assert set(REASON_CODES) == {
            "estimator_mismatch",
            "plant_logic",
            "replay_nondeterminism",
            "agent_blocked_earlier",
        }

    def test_a_second_run_that_disagrees_outranks_every_other_reason(self) -> None:
        code, why = classify_unconfirmed(
            f1_props=("P1",), f1_labels=("C1",), f2_repairs=0, nondeterministic=True
        )
        assert code == "replay_nondeterminism" and "different result" in why

    def test_the_properties_that_need_work_in_progress_are_p1_p2_p3_and_p6(self) -> None:
        """P4 and P5 are about assets that are down; P6 is a person inside during a gas alarm."""
        assert JOB_PROPERTIES == {"P1", "P2", "P3", "P6"}
        assert PERCEPTION_PROPERTIES == {"P4", "P5"}

    @pytest.mark.parametrize("prop", ["P1", "P2", "P3", "P6"])
    def test_no_repair_job_at_f2_for_a_violation_that_needs_work_is_the_agent_blocking(
        self, prop: str
    ) -> None:
        code, why = classify_unconfirmed(
            f1_props=(prop,), f1_labels=("assumption_held",), f2_repairs=0, nondeterministic=False
        )
        assert code == "agent_blocked_earlier" and prop in why

    def test_a_perception_label_means_the_estimator_when_a_job_did_run(self) -> None:
        code, _ = classify_unconfirmed(
            f1_props=("P1",), f1_labels=("C1",), f2_repairs=2, nondeterministic=False
        )
        assert code == "estimator_mismatch"

    @pytest.mark.parametrize("prop", ["P4", "P5"])
    def test_a_violation_of_an_asset_property_without_job_properties_is_the_estimator(
        self, prop: str
    ) -> None:
        code, _ = classify_unconfirmed(
            f1_props=(prop,), f1_labels=("assumption_held",), f2_repairs=0, nondeterministic=False
        )
        assert code == "estimator_mismatch"

    def test_what_is_left_is_the_plant(self) -> None:
        code, why = classify_unconfirmed(
            f1_props=("P2",), f1_labels=("A_exec",), f2_repairs=3, nondeterministic=False
        )
        assert code == "plant_logic" and "3 repair job" in why

    def test_a_job_property_with_a_repair_at_f2_is_not_blamed_on_the_agent(self) -> None:
        code, _ = classify_unconfirmed(
            f1_props=("P3",), f1_labels=("assumption_held",), f2_repairs=1, nondeterministic=False
        )
        assert code != "agent_blocked_earlier"


@pytest.fixture(scope="module")
def replay(small_context: PreparedContext) -> ReplayOutcome:
    """Careless around B3 with no shield violates at the nominal vector on most plant seeds."""
    arm = ARMS["H-C0"]
    theta = nominal_theta("in")
    seed = next(
        s
        for s in range(3000, 3010)
        if any(
            v < 0
            for v in simulate_episode(
                arm, s, Scenario.decode(theta, "in"), context=small_context
            ).result.min_robustness.values()
        )
    )
    return replay_theta(arm, theta, "in", seed, context=small_context)


class TestARealReplay:
    def test_it_is_a_replay_of_the_same_arm_on_the_same_plant_seed(
        self, replay: ReplayOutcome
    ) -> None:
        assert replay.arm == "H-C0" and replay.regime == "in" and 3000 <= replay.seed < 3010
        assert replay.f1_props and replay.f2_log is not None
        assert replay.f2_log.seed == replay.seed and replay.f2_log.arm == "H-C0"

    def test_confirmation_is_a_violation_of_a_property_f1_also_violated(
        self, replay: ReplayOutcome
    ) -> None:
        assert set(replay.common) == set(replay.f1_props) & set(replay.f2_props)
        assert replay.confirmed == bool(replay.common)
        assert (replay.reason == "confirmed") == replay.confirmed

    def test_an_unconfirmed_replay_has_exactly_one_known_reason_and_a_sentence(
        self, replay: ReplayOutcome
    ) -> None:
        if not replay.confirmed:
            assert replay.reason in REASON_CODES and len(replay.evidence) > 20

    def test_the_f2_side_ran_under_the_caching_rule(self, replay: ReplayOutcome) -> None:
        assert replay.f2_log is not None
        assert replay.f2_log.run_asset_calls == replay.f2_log.expected_run_asset_calls

    def test_the_counts_come_from_the_f2_log(self, replay: ReplayOutcome) -> None:
        assert replay.f2_log is not None
        assert replay.f2_repairs == replay.f2_log.repairs_issued
        assert replay.f2_drafts == len(replay.f2_log.drafts)


def test_a_vector_that_violates_nothing_at_f1_is_refused(small_context: PreparedContext) -> None:
    # B3 alone (no hostile proposer, shield off) on the nominal vector: nothing to replay
    arm = ARMS["B3-V3"]
    with use_context(small_context):
        res = simulate_episode(
            arm, 3001, Scenario.decode(nominal_theta("in"), "in"), context=small_context
        ).result
    if any(v < 0 for v in res.min_robustness.values()):
        pytest.skip("B3 violates on this seed; the guard needs a non-violating one")
    with pytest.raises(ValueError, match="nothing to replay"):
        replay_theta(arm, np.asarray(nominal_theta("in")), "in", 3001, context=small_context)


def test_a_missing_sibling_is_reported_before_any_episode_is_run(
    monkeypatch: pytest.MonkeyPatch, small_context: PreparedContext
) -> None:
    from typing import Any

    from closed_loop.f2 import F2Unavailable, _siblings

    real = _siblings.importlib.import_module

    def broken(name: str, package: str | None = None) -> Any:
        if name == "maintenance_agent":
            raise ImportError("no maintenance_agent here")
        return real(name, package)

    monkeypatch.setattr(_siblings.importlib, "import_module", broken)
    with pytest.raises(F2Unavailable, match="maintenance_agent"):
        replay_theta(ARMS["H-C0"], nominal_theta("in"), "in", 3000, context=small_context)
