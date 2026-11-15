"""The values tuned on ``dev`` seeds, read from ``benchmarks/tuned.json`` (ADR 0003 A1; prereg 2).

Four numbers are tuned before the freeze and then hashed into the ledger: the B3 bound-rule ``k``
(days), the B2 health-index limit, the B1 calendar interval (a fraction of the class's
characteristic life) and the planner's chance-constraint level ``delta_plan``. The primary
perception variant of the planner arms (prereg section 7) is chosen on ``dev`` by the same script
and travels in the same file as ``primary_variant``, and so does the monitor mode of arm P2 and of
the C-MON verdict (prereg section 3: the mode with the highest detection rate minus false-alarm
rate on ``dev``, ties to the lower false-alarm rate), as ``monitor_mode``. Both are optional for
the arms (a file without them leaves the provisional ``"V3"`` and ``"proxy"`` in place and says
so), but ``benchmarks/run.py --freeze`` refuses a file that names neither. This module only reads
them. It tunes nothing (``scripts/tune.py`` writes the file).

When ``benchmarks/tuned.json`` does not exist the loader returns **provisional defaults** and says
so: ``TunedValues.provisional`` is true, every arm that uses a value lists it in
``EpisodeResult.provisional`` and the report layer can print it. The defaults are what the packets
measured or used on ``dev`` seeds, and none of them is the result of a search:

``k_b3`` = 7.0 days
    the bound-rule threshold of the perception study (P4a, P4b, P4c: 3.7 planned jobs and no
    unplanned failure per episode on 100 dev episodes at k = 7, and within 0.02 coverage of the
    k = 2 and k = 0.5 runs). P6a's own default is the second bin edge, 10 days, untuned.
``b2_limit`` = 5.5
    the midpoint of the plant's two alert levels (P6a: 2 unplanned failures of 8 dev episodes).
``b1_fraction`` = 0.5
    half the characteristic life of the class (P6a: 6 unplanned failures of 8 dev episodes).
``delta_plan`` = 0.02
    the pre-registered chance-constraint level (prereg 2; P8b's default).

File format (JSON, validated here because it is read from disk, 0002 rule 6)::

    {"k_b3": 7.0, "b2_limit": 5.5, "b1_fraction": 0.5, "delta_plan": 0.02,
     "primary_variant": "V3", "monitor_mode": "proxy",
     "tuned_on": "dev seeds 1000-1099, 100 episodes per cell", "script": "..."}

``tuned_on`` and ``script`` are free text and required: a number with no provenance is refused.
Unknown keys are refused, so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from closed_loop.audit.prereg import file_sha256
from closed_loop.eval.paths import relative_path

__all__ = [
    "MONITOR_MODE_NAME",
    "PRIMARY_VARIANT_NAME",
    "PROVISIONAL",
    "TUNABLE",
    "TUNED_PATH",
    "TunedError",
    "TunedValues",
    "current",
    "load_tuned",
    "use_tuned",
]

TUNED_PATH: Final[Path] = Path(__file__).resolve().parents[2] / "benchmarks" / "tuned.json"
TUNABLE: Final[tuple[str, ...]] = ("k_b3", "b2_limit", "b1_fraction", "delta_plan")
PRIMARY_VARIANT_NAME: Final[str] = "primary_variant"  # tuned too, but a name, not a number
MONITOR_MODE_NAME: Final[str] = "monitor_mode"
_KNOWN_NAMES: Final[tuple[str, ...]] = (*TUNABLE, PRIMARY_VARIANT_NAME, MONITOR_MODE_NAME)
_MODES: Final[tuple[str, ...]] = ("opt", "pess", "inf", "proxy")
_VARIANTS: Final[tuple[str, ...]] = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V2L", "V3L", "V7L")


class TunedError(Exception):
    """``tuned.json`` exists and is malformed, or a requested name is not tunable."""


@dataclass(frozen=True)
class TunedValues:
    """The four tuned numbers and where they came from."""

    k_b3: float
    b2_limit: float
    b1_fraction: float
    delta_plan: float
    provisional: bool
    source: str  # "provisional defaults" or the path of the file read
    sha256: str | None = None  # of the normalised bytes of the file; None when provisional
    tuned_on: str = ""
    script: str = ""
    primary_variant: str = "V3"  # the variant the planner arms use (prereg section 7)
    variant_tuned: bool = False  # True when the file named the variant
    monitor_mode: str = "proxy"  # the mode that acts in P2 and decides C-MON (prereg section 3)
    mode_tuned: bool = False  # True when the file named the mode

    def value(self, name: str) -> float:
        if name not in TUNABLE:
            raise TunedError(f"{name!r} is not a tuned value; expected one of {TUNABLE}")
        return float(getattr(self, name))

    def provisional_names(self, used: tuple[str, ...]) -> tuple[str, ...]:
        """The names among ``used`` that are provisional (all of them when the file is absent)."""
        for name in used:
            if name not in _KNOWN_NAMES:
                raise TunedError(f"{name!r} is not a tuned value; expected one of {_KNOWN_NAMES}")
        return tuple(
            name
            for name in used
            if self.provisional
            or (name == PRIMARY_VARIANT_NAME and not self.variant_tuned)
            or (name == MONITOR_MODE_NAME and not self.mode_tuned)
        )


PROVISIONAL: Final[TunedValues] = TunedValues(
    k_b3=7.0,
    b2_limit=5.5,
    b1_fraction=0.5,
    delta_plan=0.02,
    provisional=True,
    source="provisional defaults (benchmarks/tuned.json absent)",
    tuned_on="dev seeds, packets P4a/P6a/P8b; not tuned",
    script="closed_loop.eval.tuned",
)


class _File(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    k_b3: float = Field(gt=0.0, le=90.0)
    b2_limit: float = Field(gt=0.0, le=20.0)
    b1_fraction: float = Field(gt=0.0, le=5.0)
    delta_plan: float = Field(gt=0.0, lt=1.0)
    primary_variant: str | None = None
    monitor_mode: str | None = None
    tuned_on: str = Field(min_length=1)
    script: str = Field(min_length=1)


def load_tuned(path: Path | None = None) -> TunedValues:
    """The tuned values in ``path`` (default ``benchmarks/tuned.json``), else the provisional ones.

    A missing file is not an error: it returns ``PROVISIONAL``. A file that exists and does not
    validate raises ``TunedError``.
    """
    target = Path(path) if path is not None else TUNED_PATH
    if not target.exists():
        return PROVISIONAL
    try:
        doc = _File.model_validate(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        raise TunedError(f"{target} is malformed: {exc}") from exc
    if doc.primary_variant is not None and doc.primary_variant not in _VARIANTS:
        raise TunedError(f"{target}: primary_variant must be one of {_VARIANTS}")
    if doc.monitor_mode is not None and doc.monitor_mode not in _MODES:
        raise TunedError(f"{target}: monitor_mode must be one of {_MODES}")
    return TunedValues(
        k_b3=doc.k_b3,
        b2_limit=doc.b2_limit,
        b1_fraction=doc.b1_fraction,
        delta_plan=doc.delta_plan,
        provisional=False,
        source=relative_path(target),  # never this machine's path (P12g, G5)
        sha256=file_sha256(target),
        tuned_on=doc.tuned_on,
        script=doc.script,
        primary_variant=doc.primary_variant or PROVISIONAL.primary_variant,
        variant_tuned=doc.primary_variant is not None,
        monitor_mode=doc.monitor_mode or PROVISIONAL.monitor_mode,
        mode_tuned=doc.monitor_mode is not None,
    )


_override: TunedValues | None = None
_cached: TunedValues | None = None


def current() -> TunedValues:
    """The values the arms use: the file (read once per process) or an override of ``use_tuned``."""
    global _cached
    if _override is not None:
        return _override
    if _cached is None:
        _cached = load_tuned()
    return _cached


class use_tuned:  # noqa: N801 - used as a context manager, like unittest.mock.patch
    """Context manager: use ``values``, or the current values with ``changes`` applied, inside."""

    def __init__(self, values: TunedValues | None = None, **changes: Any) -> None:
        base = values if values is not None else current()
        self._values = replace(base, **changes) if changes else base

    def __enter__(self) -> TunedValues:
        global _override
        self._previous = _override
        _override = self._values
        return self._values

    def __exit__(self, *exc: object) -> None:
        global _override
        _override = self._previous
