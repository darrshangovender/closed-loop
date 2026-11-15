"""The ``Proposer`` Protocol (frozen: docs/adr/0002-interfaces.md, section 3.8).

Every controller of the SYNTHETIC plant, honest or hostile, is a ``Proposer``: it sees an
``Observation`` and returns commands. It never sees truth. ``Observation`` carries no truth field
(``tests/plant`` checks the field names by reflection, and ``tests/proposers`` checks that no
proposer reads one), so a proposer is as blind as a controller on a real site: the health index, the
perception layer's interval and set, the open work order, the shift-handover note.

The shield (``closed_loop.shield.runtime.ShieldedProposer``) and the interlock
(``closed_loop.eval.interlock.InterlockedProposer``) wrap a ``Proposer`` and are ``Proposer``-shaped
themselves, which is what lets an arm swap one for the other.
"""

from __future__ import annotations

from typing import Protocol

from closed_loop.core.types import Command
from closed_loop.plant.plant import Observation


class Proposer(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def propose(self, obs: Observation) -> list[Command]: ...


__all__ = ["Proposer"]
