"""``LLMProposer``: an optional adapter that asks a language model for the next commands (P6b).

**It is never run in the tests or the benchmarks.** Constructing it raises ``NotConfigured`` unless
both of these hold: the environment variable ``CLOSED_LOOP_LLM`` is ``1`` and the ``anthropic``
package is installed. ``pyproject.toml`` has no extra for it yet (a packaging change that belongs to
the owner of that file), so by default the second condition fails as well. A model name is also
needed (the ``model`` keyword or ``CLOSED_LOOP_LLM_MODEL``); none is hard-coded here.

Constructing it does not build a client and does not touch the network; the client is created on
the first ``propose``. The suite proves it never gets that far: ``tests/conftest.py`` blocks every
non-local socket, and no test calls ``propose`` on a configured instance.

What is here and testable without a model: ``render_observation`` (the text a model would be shown,
which carries only what ``Observation`` carries: no truth) and ``parse_commands`` (a model's reply
to commands, anything outside the job alphabet dropped). The call itself, in ``_complete``, is the
public ``messages.create`` interface of the SDK and has not been run against the service.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Any

from closed_loop.core.types import Cmd, Command, PlanCmd
from closed_loop.plant.jobs import ALPHABET
from closed_loop.plant.plant import Observation

__all__ = ["ENV_FLAG", "ENV_MODEL", "LLMProposer", "NotConfigured", "parse_commands",
           "render_observation"]  # fmt: skip

ENV_FLAG = "CLOSED_LOOP_LLM"
ENV_MODEL = "CLOSED_LOOP_LLM_MODEL"
SDK_MODULE = "anthropic"

_SYSTEM = (
    "You control a SYNTHETIC mine-site plant. Reply with at most one command per line, in the "
    "form 'ASSET COMMAND' (for example 'CR-501 LOCK') or 'ASSET RAISE_WO MODE'. Reply with "
    "nothing if no command is needed."
)


class NotConfigured(RuntimeError):
    """The LLM adapter was asked for without the flag, the package or a model."""


def render_observation(obs: Observation) -> str:
    """The observation as text: what a controller on site could read, and nothing else."""
    lines = [f"quarter {obs.quarter}, day {obs.day}, shift {obs.shift}"]
    for aid, v in obs.assets.items():
        job = f", job {v.job.status} ({v.job.mode})" if v.job is not None else ""
        rul = "" if v.rul_lo is None else f", RUL lower bound {v.rul_lo:.1f} d"
        lines.append(
            f"{aid}: op {v.op.value}, permit {v.permit.value}, occupied {v.occupied}, "
            f"last gas test {v.gas_age_q} quarters ago, alarm {v.alarm}, handover next {v.soon}"
            f"{rul}{job}"
        )
    if obs.note is not None:
        lines.append(f"handover note: {obs.note}")
    return "\n".join(lines)


def parse_commands(reply: str) -> list[Command]:
    """``ASSET COMMAND`` and ``ASSET RAISE_WO MODE`` lines to commands; anything else is dropped."""
    out: list[Command] = []
    for line in reply.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        asset_id, name = parts[0], parts[1].upper()
        if name == PlanCmd.RAISE_WO.value and len(parts) >= 3:
            out.append(Command(asset_id, PlanCmd.RAISE_WO, parts[2].upper(), "llm"))
        elif name in {c.value for c in ALPHABET}:
            out.append(Command(asset_id, Cmd(name), None, "llm"))
    return out


class LLMProposer:
    """Raises ``NotConfigured`` unless ``CLOSED_LOOP_LLM=1``, the SDK is there and a model set."""

    name = "llm"

    def __init__(self, *args: Any, **kw: Any) -> None:
        if os.environ.get(ENV_FLAG) != "1":
            raise NotConfigured(
                f"set {ENV_FLAG}=1 to use the LLM adapter (it is never run in tests)"
            )
        if importlib.util.find_spec(SDK_MODULE) is None:
            raise NotConfigured(f"the {SDK_MODULE!r} package is not installed")
        model = kw.pop("model", None) or os.environ.get(ENV_MODEL)
        if not model:
            raise NotConfigured(f"give model=... or set {ENV_MODEL}")
        self.model: str = model
        self.max_tokens: int = int(kw.pop("max_tokens", 256))
        self._client: Any = None

    def reset(self, seed: int) -> None:
        pass

    def propose(self, obs: Observation) -> list[Command]:
        return parse_commands(self._complete(render_observation(obs)))

    def _complete(self, prompt: str) -> str:
        if self._client is None:
            self._client = importlib.import_module(SDK_MODULE).Anthropic()
        reply = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(block, "text", "") for block in reply.content)
