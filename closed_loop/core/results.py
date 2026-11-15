"""``ResultRecord``: the only form in which a metric is published (0002, section 1, rule 7).

Interface text: docs/adr/0002-interfaces.md, section 3.2.
"""

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from closed_loop.core.types import Fidelity


@dataclass(frozen=True)
class ResultRecord:
    """One measured number, with what it was measured on and the script that produced it.

    ``extra`` is copied on construction, so a record does not alias the caller's dict. Records
    compare by value but are not hashable while ``extra`` is a dict.
    """

    name: str
    value: float
    n: int
    fidelity: Fidelity
    measured_on: str  # e.g. "held-out seeds, nominal condition, 500 episodes"
    script: str  # module that produced it
    ci_lo: float | None = None
    ci_hi: float | None = None
    ci_kind: str = ""  # "confseq-0.05"
    extra: Mapping[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ResultRecord.name is empty")
        if not self.measured_on.strip():
            raise ValueError(f"ResultRecord {self.name!r}: measured_on is empty")
        if not self.script.strip():
            raise ValueError(f"ResultRecord {self.name!r}: script is empty")
        # Frozen dataclass: coerce through object.__setattr__ so numpy scalars become plain
        # Python numbers (they must survive json) and a bad fidelity string fails here.
        value = float(self.value)
        if math.isnan(value):
            raise ValueError(f"ResultRecord {self.name!r}: value is NaN")
        object.__setattr__(self, "value", value)
        n = operator.index(self.n)
        if n < 0:
            raise ValueError(f"ResultRecord {self.name!r}: n is negative ({n})")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "fidelity", Fidelity(self.fidelity))
        lo = None if self.ci_lo is None else float(self.ci_lo)
        hi = None if self.ci_hi is None else float(self.ci_hi)
        for label, bound in (("ci_lo", lo), ("ci_hi", hi)):
            if bound is not None and math.isnan(bound):
                raise ValueError(f"ResultRecord {self.name!r}: {label} is NaN")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"ResultRecord {self.name!r}: ci_lo {lo} > ci_hi {hi}")
        object.__setattr__(self, "ci_lo", lo)
        object.__setattr__(self, "ci_hi", hi)
        object.__setattr__(self, "extra", dict(self.extra))

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python form for ``results.json``; ``from_dict`` is its exact inverse."""
        return {
            "name": self.name,
            "value": self.value,
            "n": self.n,
            "fidelity": self.fidelity.value,
            "measured_on": self.measured_on,
            "script": self.script,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "ci_kind": self.ci_kind,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            name=data["name"],
            value=data["value"],
            n=data["n"],
            fidelity=Fidelity(data["fidelity"]),
            measured_on=data["measured_on"],
            script=data["script"],
            ci_lo=data.get("ci_lo"),
            ci_hi=data.get("ci_hi"),
            ci_kind=data.get("ci_kind", ""),
            extra=data.get("extra", {}),
        )
