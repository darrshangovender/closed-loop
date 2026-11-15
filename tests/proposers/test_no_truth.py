"""Proposers see ``Observation`` only: no truth field is available to them, and none is read.

Three layers. (1) The observation types carry no truth-named field. (2) The proposer modules do not
name any truth identifier or import the truth type. (3) A whole episode of every proposer runs with
the plant's truth made unreadable: any access would raise.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from closed_loop.plant import plant as plant_module
from closed_loop.plant.des import JobView
from closed_loop.plant.notes import cached_scan
from closed_loop.plant.observe import DailyObs
from closed_loop.plant.plant import AssetView, F1Plant, Observation
from closed_loop.proposers import Careless, Gullible, hostile, llm, make_baseline
from closed_loop.proposers import base as base_module
from closed_loop.proposers import baselines as baselines_module
from closed_loop.proposers import restart as restart_module
from closed_loop.proposers import stall as stall_module
from tests.proposers._loop import run_episode

TRUTH_WORDS = (
    "truth",
    "TruthView",
    "true_rul",
    "health_bin",
    "failure_time",
    "life_id",
    "life_mode",
    "rate_rul",
    "realised",
    "alert_truth",
    "AlertTruth",
    "uncontrolled_failure",
)
PROPOSER_MODULES = [base_module, baselines_module, hostile, llm, restart_module, stall_module]


def field_names(cls: type) -> list[str]:
    return [f.name for f in dataclasses.fields(cls)]


class TestTheObservationHasNoTruth:
    @pytest.mark.parametrize("cls", [Observation, AssetView, JobView, DailyObs])
    def test_no_field_names_a_truth(self, cls: type) -> None:
        for name in field_names(cls):
            for word in ("true", "truth", "health_bin", "failure_time", "life_id", "realised"):
                assert word not in name.lower(), (cls.__name__, name)

    def test_the_truth_is_on_the_plant_not_on_the_observation(self) -> None:
        assert hasattr(F1Plant, "truth")
        assert "truth" not in field_names(Observation)


class TestTheProposerModulesNameNoTruth:
    @pytest.mark.parametrize("module", PROPOSER_MODULES, ids=lambda m: m.__name__)
    def test_source(self, module: object) -> None:
        source = inspect.getsource(module)  # type: ignore[arg-type]
        # comments and docstrings may say "no truth"; look at code identifiers and imports only
        code = re.sub(r'""".*?"""', "", source, flags=re.S)
        code = re.sub(r"#.*", "", code)
        for word in TRUTH_WORDS:
            assert not re.search(rf"\b{word}\b", code), (module.__name__, word)  # type: ignore[attr-defined]

    @pytest.mark.parametrize("module", PROPOSER_MODULES, ids=lambda m: m.__name__)
    def test_the_modules_import_no_truth_type(self, module: object) -> None:
        for name, value in vars(module).items():
            assert value is not plant_module.TruthView, (module.__name__, name)  # type: ignore[attr-defined]
            assert value is not plant_module.AlertTruth, (module.__name__, name)  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: make_baseline("B0"),
            lambda: make_baseline("B3", diagnoser=None),
            lambda: Careless(make_baseline("B2")),
            lambda: Gullible(make_baseline("B2"), True, cached_scan),
        ],
    )
    def test_propose_takes_the_observation_and_nothing_else(self, factory) -> None:  # type: ignore[no-untyped-def]
        sig = inspect.signature(factory().propose)
        assert list(sig.parameters) == ["obs"]


class TestEpisodesWithTheTruthUnreadable:
    @pytest.fixture
    def tripwire(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(self: F1Plant) -> object:
            raise AssertionError("a proposer read the plant's truth")

        monkeypatch.setattr(F1Plant, "truth", property(boom))

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: make_baseline("B0"),
            lambda: make_baseline("B1"),
            lambda: make_baseline("B2"),
            lambda: Careless(make_baseline("B2")),
            lambda: Gullible(make_baseline("B2"), False, cached_scan),
        ],
        ids=["B0", "B1", "B2", "careless", "gullible"],
    )
    def test_a_whole_episode_never_touches_it(self, tripwire: None, factory) -> None:  # type: ignore[no-untyped-def]
        ep = run_episode(factory(), 2, horizon_q=96 * 30)
        assert ep.trace is not None
