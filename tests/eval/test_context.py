"""``eval.context``: fitted once per process, picklable, rebuildable, seed roles respected."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from closed_loop.eval import context as context_module
from closed_loop.eval.context import (
    CLAUSE_AS_OF,
    STALE_AS_OF,
    ContextSpec,
    PreparedContext,
    current_context,
    install_context,
    prepare_context,
    shield_kit,
    use_context,
)
from closed_loop.shield.table import ShieldTable
from tests.eval.conftest import SMALL_SPEC


class TestPreparation:
    def test_it_holds_what_an_episode_needs(self, small_context: PreparedContext) -> None:
        ctx = small_context
        dev = ctx.dev
        assert dev.scores.size and dev.prior_scores.size and dev.life_scores.shape[1] == 4
        assert set(ctx.diagnosers) == {"scm", "corr"}
        assert all(d.fitted for d in ctx.diagnosers.values())  # type: ignore[attr-defined]
        assert ctx.proxy_band.alpha == pytest.approx(0.05)
        assert ctx.world_fit and ctx.built_s > 0
        assert set(ctx.timings_s) == {
            "conformal_pools",
            "proxy_band",
            "diagnosers",
            "world_model",
            "shield_tables",
        }

    def test_the_two_specifications_are_rev_3_and_rev_2(
        self, small_context: PreparedContext
    ) -> None:
        assert small_context.spec_clause.retest_q == 2
        assert small_context.spec_stale.retest_q == 8
        assert CLAUSE_AS_OF.isoformat() == "2026-10-01"
        assert STALE_AS_OF.isoformat() == "2026-01-01"
        assert small_context.spec_for_shield("clause") is small_context.spec_clause
        with pytest.raises(ValueError):
            small_context.spec_for_shield("rev1")

    def test_preparation_is_deterministic(self, small_context: PreparedContext) -> None:
        again = prepare_context(SMALL_SPEC)
        assert np.array_equal(again.dev.scores, small_context.dev.scores)
        assert np.array_equal(again.dev.life_scores, small_context.dev.life_scores)
        assert again.proxy_band == small_context.proxy_band
        assert dict(again.world_fit) == dict(small_context.world_fit)
        assert dict(again.shield_json) == dict(small_context.shield_json)

    def test_calibration_uses_calib_seeds_only(self) -> None:
        for bad in (
            ContextSpec(conformal_seeds=(1000, 1010)),  # dev, not calib
            ContextSpec(proxy_seeds=(3000, 3010)),  # search
            ContextSpec(diagnoser_seeds=(2990, 3010)),  # runs into the search role
        ):
            with pytest.raises(ValueError):
                prepare_context(bad)

    def test_the_spec_names_what_the_default_is_built_from(self) -> None:
        spec = ContextSpec()
        assert spec.seeds("conformal_seeds") == range(2000, 2300)
        assert spec.seeds("diagnoser_seeds") == range(2300, 2375)  # 75 seeds x 4 lives
        assert spec.seeds("world_model_seeds") == range(1000, 1100)  # dev, the planner's own rule


class TestPickling:
    def test_it_round_trips(self, small_context: PreparedContext) -> None:
        copy = pickle.loads(pickle.dumps(small_context))
        assert np.array_equal(copy.dev.scores, small_context.dev.scores)
        assert set(copy.diagnosers) == {"scm", "corr"}
        assert dict(copy.shield_json) == dict(small_context.shield_json)
        assert copy.spec == small_context.spec

    def test_the_tables_are_rebuilt_from_text_not_pickled(
        self, small_context: PreparedContext
    ) -> None:
        assert all(isinstance(v, str) for v in small_context.shield_json.values())
        kit = shield_kit(small_context, "clause")
        assert isinstance(kit.crusher, ShieldTable) and isinstance(kit.pumps, ShieldTable)
        again = shield_kit(pickle.loads(pickle.dumps(small_context)), "clause")
        assert again is kit  # keyed by the text's hash: one build per process
        assert again.crusher.sha256() == kit.crusher.sha256()


class TestShieldKit:
    def test_the_stale_kit_is_the_two_hour_game(self, small_context: PreparedContext) -> None:
        clause, stale = shield_kit(small_context, "clause"), shield_kit(small_context, "stale")
        assert clause.mapper.retest_q == 2 and stale.mapper.retest_q == 8
        assert clause.crusher.sha256() != stale.crusher.sha256()
        assert clause.pumps.sha256() == stale.pumps.sha256()  # the pump pair has no clause
        assert clause.crusher_model.parameters() != stale.crusher_model.parameters()


class TestCurrentContext:
    def test_use_context_installs_it_and_restores_the_previous(
        self, small_context: PreparedContext
    ) -> None:
        with use_context(small_context):
            assert current_context() is small_context
            other = pickle.loads(pickle.dumps(small_context))
            with use_context(other):
                assert current_context() is other
            assert current_context() is small_context

    def test_install_context_sets_the_process_default(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(context_module, "_PROCESS", {})
        install_context(small_context)
        assert current_context() is small_context
