"""Command proposers: baselines, hostile wrappers and the LLM stub (packets P6a and P6b)."""

from closed_loop.plant.notes import NOTE_BANK
from closed_loop.proposers.base import Proposer
from closed_loop.proposers.baselines import (
    BoundRule,
    Calendar,
    FixedLimit,
    RunToFailure,
    make_baseline,
)
from closed_loop.proposers.hostile import Careless, Gullible, screener_misses
from closed_loop.proposers.llm import LLMProposer, NotConfigured

__all__ = [
    "NOTE_BANK",
    "BoundRule",
    "Calendar",
    "Careless",
    "FixedLimit",
    "Gullible",
    "LLMProposer",
    "NotConfigured",
    "Proposer",
    "RunToFailure",
    "make_baseline",
    "screener_misses",
]
