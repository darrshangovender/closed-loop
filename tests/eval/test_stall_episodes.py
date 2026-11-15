"""0003 A18 R4 and R5 in whole episodes: who reaches the stalled path, and who gets an operator.

Development seeds only, with the small prepared context of ``conftest.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from closed_loop.core.types import Command
from closed_loop.eval.arms import ARMS
from closed_loop.eval.episode import scenario_for, simulate_episode
from closed_loop.proposers.restart import OPERATOR, Restarting
from closed_loop.proposers.stall import StallWatch


class TestWhoGetsAnOperator:
    """The restart wrapper is built for an arm with a shield or an interlock and for no other."""

    @pytest.mark.parametrize(
        ("arm", "wanted", "pair"),
        [
            ("B2", False, False),
            ("P0", False, False),
            ("B3-S", True, True),
            ("P1", True, True),
            ("P1-nopair", True, False),  # R8: the arm that keeps the deadlock measured
            ("H-C-I", True, False),  # R8: not for an interlock (rule R_PUMP_LOCK wants the pump)
            ("P1-I", True, False),
        ],
    )
    def test_the_wrapper_is_in_the_chain_exactly_where_the_ruling_puts_it(
        self,
        arm: str,
        wanted: bool,
        pair: bool,
        small_context: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        built: list[bool] = []
        real_init = Restarting.__init__

        def init(self: Restarting, inner: Any, pair_stop: bool = False) -> None:
            built.append(pair_stop)
            real_init(self, inner, pair_stop)

        monkeypatch.setattr(Restarting, "__init__", init)
        simulate_episode(ARMS[arm], 1103, scenario_for("nominal"), context=small_context)
        assert bool(built) is wanted
        if wanted:
            assert built == [pair]


class TestInWholeEpisodes:
    @staticmethod
    def spy(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[int]]:
        stalled: list[Any] = []
        offers: list[int] = []
        real_stalled = StallWatch.stalled
        real_propose = Restarting.propose

        def watching(self: StallWatch, aid: str, obs: Any) -> bool:
            hit = real_stalled(self, aid, obs)
            if hit:
                stalled.append((aid, obs.quarter))
            return hit

        def counting(self: Restarting, obs: Any) -> list[Command]:
            out = real_propose(self, obs)
            offers.append(sum(1 for c in out if c.source == OPERATOR))
            return out

        monkeypatch.setattr(StallWatch, "stalled", watching)
        monkeypatch.setattr(Restarting, "propose", counting)
        return stalled, offers

    @pytest.mark.parametrize("arm", ["B2", "B3-V2L", "P0"])
    def test_an_arm_without_a_shield_or_interlock_never_reaches_the_stalled_path(
        self, arm: str, small_context: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing stops an asset except a job, and a job closes with its own START, so the stall
        watch must never fire for these arms, and no operator is put in front of them."""
        stalled, offers = self.spy(monkeypatch)
        assert not ARMS[arm].shield and not ARMS[arm].interlock
        simulate_episode(ARMS[arm], 1103, scenario_for("nominal"), context=small_context)
        assert stalled == [] and offers == []

    @pytest.mark.parametrize("arm", ["B3-S", "P1"])
    def test_a_shielded_arm_has_an_operator_and_a_stopped_asset_does_get_a_job(
        self, arm: str, small_context: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shield stops assets on a bound (seed 1101 stalled B3-S before A18); the operator
        offers a restart each day, and once the stop has stood long enough the asset gets a job."""
        stalled, offers = self.spy(monkeypatch)
        rec = simulate_episode(ARMS[arm], 1101, scenario_for("nominal"), context=small_context)
        assert offers, "the operator was never in the chain"
        assert sum(offers) > 0, "no stopped asset was ever offered a restart"
        assert stalled, "no asset stood stopped long enough to be a stalled one"
        assert rec.result.planned_jobs + rec.result.unplanned_failures > 0
