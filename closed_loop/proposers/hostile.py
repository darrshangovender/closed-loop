"""Hostile proposers (packet P6b): the careless fitter and the gullible one.

Both wrap an honest ``Proposer`` and corrupt what it does, in the two ways 0001 section 8 names.
They exist to give the shield something to stop; with the shield off they produce the violations the
properties describe. Neither reads truth: they see the ``Observation`` the inner proposer sees.

``Careless``
    each quarter, for each command the inner proposer issues from the job alphabet, with probability
    ``p_skip`` the fitter does something else out of order. The rule is ``SKIP_TO``: a step that
    comes before work starts (stop, lock, prove, issue) becomes ``BEGIN`` ("BEGIN before PROVE");
    a step that precedes a re-entry (revalidate, test) becomes ``ENTER``; ``EXIT`` is forgotten (no
    command); finishing steps become ``START``. The draw is one keyed uniform per asset and quarter
    (``rng_for(seed, "careless", asset_id)``), so which quarters go wrong does not depend on what
    the inner proposer happens to do, and a paired arm (the same seed with and without the
    shield) sees the same carelessness.

``Gullible``
    reads the shift-handover note. A note that carries an instruction (the bank's ``induces`` tag:
    a plain instruction, one addressed to the assistant, one with hidden formatting; English or
    Arabic) makes it issue the instructed command for the asset the note names, in place of whatever
    the inner proposer wanted for that asset in that quarter. The tag stands for the gullible
    reader's understanding: it is assumed to read every note in the bank, in either language, and a
    text outside the bank is read as routine. With ``screen=True`` a note that ``scan`` flags (a
    rule name comes back) is dropped before it is read; the real screener is
    ``plant_copilot.injection.scan_text``, the cached copy of its verdicts is ``notes.cached_scan``.

The screener's miss list (malicious notes that ``scan`` lets through) is an output:
``screener_misses(scan)``. A miss is a note the gullible reader obeys. Nothing here is a claim that
the screener is bad or good; the counts are whatever they are.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.rng import rng_for
from closed_loop.core.timegrid import HORIZON_Q
from closed_loop.core.types import Cmd, Command
from closed_loop.plant.notes import NOTE_BANK, NoteTemplate, note_template
from closed_loop.plant.plant import Observation
from closed_loop.proposers.base import Proposer

__all__ = [
    "SKIP_TO",
    "Careless",
    "Gullible",
    "ObeyedNote",
    "Skip",
    "screener_misses",
]

# The command a careless fitter issues when the job runner's command is the key. None: forgotten.
# ENTER is not in the table: it is already the step the shortcuts jump to.
SKIP_TO: Final[dict[Cmd, Cmd | None]] = {
    Cmd.STOP: Cmd.BEGIN,
    Cmd.LOCK: Cmd.BEGIN,
    Cmd.PROVE: Cmd.BEGIN,
    Cmd.ISSUE: Cmd.BEGIN,
    Cmd.REVAL: Cmd.ENTER,
    Cmd.BEGIN: Cmd.ENTER,
    Cmd.TEST: Cmd.ENTER,
    Cmd.EXIT: None,
    Cmd.COMPLETE: Cmd.START,
    Cmd.CLOSE: Cmd.START,
    Cmd.UNISO: Cmd.START,
    Cmd.START: Cmd.START,
}


@dataclass(frozen=True)
class Skip:
    """One out-of-order moment of a careless fitter."""

    quarter: int
    asset_id: str
    intended: Cmd
    issued: Cmd | None  # None: the command was forgotten


class Careless:
    """Wraps ``inner``; see the module docstring."""

    def __init__(self, inner: Proposer, p_skip: float = 0.05) -> None:
        if not 0.0 <= p_skip <= 1.0:
            raise ValueError(f"p_skip must be in [0, 1], got {p_skip}")
        self.inner = inner
        self.p_skip = p_skip
        self.name = f"careless({inner.name})"
        self.seed = 0
        self.skips: list[Skip] = []
        self._u: dict[str, NDArray[np.float64]] = {}

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)
        self.seed = seed
        self.skips = []
        self._u = {}

    def _draw(self, asset_id: str, quarter: int) -> float:
        u = self._u.get(asset_id)
        if u is None:
            u = rng_for(self.seed, "careless", asset_id).random(HORIZON_Q)
            self._u[asset_id] = u
        return float(u[quarter % HORIZON_Q])

    def propose(self, obs: Observation) -> list[Command]:
        out: list[Command] = []
        for c in self.inner.propose(obs):
            cmd = c.cmd
            if (
                isinstance(cmd, Cmd)
                and cmd in SKIP_TO
                and self._draw(c.asset_id, obs.quarter) < self.p_skip
            ):
                target = SKIP_TO[cmd]
                if target is not cmd:
                    self.skips.append(Skip(obs.quarter, c.asset_id, cmd, target))
                    if target is not None:
                        out.append(Command(c.asset_id, target, None, self.name))
                    continue
            out.append(c)
        return out


@dataclass(frozen=True)
class ObeyedNote:
    """A note the gullible reader acted on."""

    quarter: int
    note_id: str
    asset_id: str
    command: Cmd


class Gullible:
    """Wraps ``inner``; see the module docstring."""

    def __init__(self, inner: Proposer, screen: bool, scan: Callable[[str], str | None]) -> None:
        self.inner = inner
        self.screen = screen
        self.scan = scan
        self.name = f"gullible({inner.name}{',screened' if screen else ''})"
        self.notes_seen = 0
        self.notes_dropped = 0
        self.obeyed: list[ObeyedNote] = []
        self.dropped: list[tuple[int, str, str]] = []  # (quarter, note id, rule that flagged it)

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)
        self.notes_seen = 0
        self.notes_dropped = 0
        self.obeyed = []
        self.dropped = []

    def propose(self, obs: Observation) -> list[Command]:
        commands = self.inner.propose(obs)
        note = obs.note
        if note is None:
            return commands
        self.notes_seen += 1
        template = note_template(note)
        if self.screen:
            rule = self.scan(note)
            if rule is not None:
                self.notes_dropped += 1
                self.dropped.append((obs.quarter, template.note_id if template else "", rule))
                return commands
        if template is None or template.induces is None or template.asset_id is None:
            return commands  # routine, or not in the bank: read as routine
        asset_id, cmd = template.asset_id, template.induces
        self.obeyed.append(ObeyedNote(obs.quarter, template.note_id, asset_id, cmd))
        kept = [c for c in commands if c.asset_id != asset_id]
        return [*kept, Command(asset_id, cmd, None, self.name)]


def screener_misses(scan: Callable[[str], str | None]) -> tuple[NoteTemplate, ...]:
    """Malicious notes of the bank that ``scan`` does not flag, in bank order."""
    return tuple(n for n in NOTE_BANK if n.malicious and scan(n.text) is None)
