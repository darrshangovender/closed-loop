"""The LLM adapter (packet P6b): it refuses to be built by default, and no test builds a client.

Nothing here constructs a configured ``LLMProposer``: each test that sets the flag expects the
constructor to refuse before any client could exist. ``tests/conftest.py`` blocks the network for
the whole session, and ``test_network_blocked.py`` proves the block works.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from closed_loop.core.types import Cmd, Command, PlanCmd
from closed_loop.proposers import LLMProposer, NotConfigured
from closed_loop.proposers.llm import ENV_FLAG, ENV_MODEL, parse_commands, render_observation
from tests.proposers._views import job, observation, view


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.delenv(ENV_MODEL, raising=False)


class TestNotConfigured:
    def test_by_default_it_raises(self) -> None:
        with pytest.raises(NotConfigured, match=ENV_FLAG):
            LLMProposer()

    def test_arguments_do_not_configure_it(self) -> None:
        with pytest.raises(NotConfigured):
            LLMProposer("a prompt", model="some-model", max_tokens=10)

    @pytest.mark.parametrize("value", ["", "0", "true", "yes", "2", " 1"])
    def test_only_the_exact_flag_counts(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(ENV_FLAG, value)
        with pytest.raises(NotConfigured, match=ENV_FLAG):
            LLMProposer(model="some-model")

    def test_with_the_flag_but_without_the_package_it_still_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_FLAG, "1")
        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: None if name == "anthropic" else real(name, *a),
        )
        with pytest.raises(NotConfigured, match="not installed"):
            LLMProposer(model="some-model")

    def test_with_the_flag_and_the_package_but_no_model_it_still_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_FLAG, "1")
        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: (
                importlib.machinery.ModuleSpec("anthropic", None)
                if name == "anthropic"
                else real(name, *a)
            ),
        )
        with pytest.raises(NotConfigured, match="model"):
            LLMProposer()

    def test_not_configured_is_a_runtime_error(self) -> None:
        assert issubclass(NotConfigured, RuntimeError)


class TestNoClientIsBuilt:
    def test_importing_the_proposers_imports_no_sdk_no_http_library_and_no_sibling(self) -> None:
        code = "\n".join(
            [
                "import sys",
                "import closed_loop.proposers",
                "banned = ('anthropic', 'openai', 'httpx', 'requests', 'urllib3', 'aiohttp',",
                "          'asset_health', 'maintenance_agent', 'plant_copilot',",
                "          'pandas', 'sklearn')",
                "print(','.join(m for m in banned if m in sys.modules))",
            ]
        )
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
        )
        assert out.stdout.strip() == ""

    def test_the_module_calls_no_network_function_outside_the_complete_method(self) -> None:
        import inspect

        from closed_loop.proposers import llm

        source = inspect.getsource(llm)
        before_complete = source.split("def _complete")[0]
        assert "import_module(SDK_MODULE)" not in before_complete
        for statement in ("import socket", "import urllib", "import requests", "import httpx"):
            assert statement not in source, statement


class TestRenderObservation:
    def test_it_shows_what_the_observation_carries_and_nothing_more(self) -> None:
        o = observation(
            [view("CR-501", hi_last=6.0, rul_lo=12.5, job_view=job("CR-501", mode="LINER_WEAR"))],
            quarter=40,
            note="Handover note.",
        )
        text = render_observation(o)
        assert "quarter 40" in text and "CR-501" in text and "RUL lower bound 12.5" in text
        assert "LINER_WEAR" in text and "handover note: Handover note." in text

    def test_no_truth_vocabulary_reaches_the_prompt(self) -> None:
        text = render_observation(observation([view("PU-201", hi_last=3.0, rul_lo=30.0)]))
        for word in ("true", "truth", "health_bin", "failure_time", "life_id"):
            assert word not in text.lower()


class TestParseCommands:
    def test_commands_and_work_orders(self) -> None:
        got = parse_commands("CR-501 lock\nPU-201 RAISE_WO impeller_wear\nCV-101 START")
        assert got == [
            Command("CR-501", Cmd.LOCK, None, "llm"),
            Command("PU-201", PlanCmd.RAISE_WO, "IMPELLER_WEAR", "llm"),
            Command("CV-101", Cmd.START, None, "llm"),
        ]

    @pytest.mark.parametrize(
        "line",
        ["", "CR-501", "CR-501 DERATE", "CR-501 NOOP", "CR-501 INSPECT", "CR-501 RAISE_WO",
         "CR-501 EXPLODE", "just some words"],
    )  # fmt: skip
    def test_anything_else_is_dropped(self, line: str) -> None:
        assert parse_commands(line) == []

    def test_a_reply_with_commentary_keeps_only_the_commands(self) -> None:
        reply = "I would stop the crusher first.\nCR-501 STOP\nThen wait."
        assert [c.cmd for c in parse_commands(reply)] == [Cmd.STOP]
