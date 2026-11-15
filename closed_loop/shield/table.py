"""The runtime shield table, built from a solved game and serialised with a SHA-256.

For an observable state ``o`` and a perception set ``B`` (a set of health bins; one interval per
component), the table holds the commands that are safe for **every** hidden health value in ``B``
that is a winning state at ``o``:

    A(o, B) = intersection over h in B with (o, h) in W of safe(o, h)

It is the maximally permissive set, not one chosen command. The shield quantifies over the whole
set, so a wider set costs throughput, not safety, provided the contract C1 holds (the true bin is
in the set) and the model's assumptions hold (see ``abstraction``). The table proves nothing
beyond the finite abstraction it was built from, and C1 is read per life (ADR 0003, A6). Health
values that are not winning at ``o`` are skipped: the proof keeps the system inside ``W``, so the
true state is never one of them while the assumptions hold.

Key layout. ``entries`` is keyed by ``(observable, health_set)``. ``health_set`` is ``(lo, hi)``
for the crusher and for the conveyor CV-101 (ADR 0003 A12.3) and ``(lo1, hi1, lo2, hi2)`` for the
pump pair: one inclusive interval per hidden component, in the order ``AbstractModel.health``
returns them. Three tables make the shield and ``build_shield`` is generic over the three games.
Entries exist for every interval of width up to ``max_width`` whose box holds at least one
winning state at ``o``. ``allowed`` also
answers wider sets, exactly, as the intersection of the entries that cover them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any

from closed_loop.core.types import N_BINS
from closed_loop.shield.abstraction import (
    ABSTRACTION_VERSION,
    AbstractModel,
)
from closed_loop.shield.game import Winning, canonical, solve

FORMAT = "closed-loop-shield-table/1"
MODEL_PREFIX = "model."
SPEC_PREFIX = "spec."

# Substitution order (ADR 0001, section 4.3): stop first, then leave, then do nothing, then the
# rest in the model's own order. A joint command ranks by its components, left to right.
_PREFERRED = ("STOP", "EXIT", "NOOP")


def _dump(x: Any) -> str:
    return json.dumps(x, separators=(",", ":"), sort_keys=True, ensure_ascii=True)


def _tuplify(x: Any) -> Any:
    return tuple(_tuplify(v) for v in x) if isinstance(x, list) else x


def default_priority(commands: Sequence[str]) -> tuple[str, ...]:
    def rank(c: str) -> tuple[tuple[int, ...], int]:
        parts = tuple(
            _PREFERRED.index(p) if p in _PREFERRED else len(_PREFERRED) for p in c.split("|")
        )
        return parts, commands.index(c)

    return tuple(sorted(commands, key=rank))


@dataclass(frozen=True)
class ShieldTable:
    model_name: str
    entries: Mapping[tuple[Hashable, tuple[int, ...]], frozenset[str]]
    priority: tuple[str, ...]
    meta: Mapping[str, str]
    # Derived in __post_init__; not part of the table's identity.
    _max_width: int = field(init=False, repr=False, compare=False, default=0)
    _dims: int = field(init=False, repr=False, compare=False, default=0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))
        object.__setattr__(self, "priority", tuple(self.priority))
        widths = [
            hi - lo + 1
            for _o, hs in self.entries
            for lo, hi in zip(hs[0::2], hs[1::2], strict=True)
        ]
        dims = {len(hs) // 2 for _o, hs in self.entries}
        if len(dims) > 1:
            raise ValueError("entries mix health sets of different sizes")
        object.__setattr__(self, "_max_width", max(widths, default=0))
        object.__setattr__(self, "_dims", dims.pop() if dims else 0)

    # ------------------------------------------------------------------ lookup

    def allowed(self, o: Hashable, health_set: Sequence[int]) -> frozenset[str]:
        """Commands safe for every winning hidden health value in ``health_set`` at ``o``.

        An empty result means no command is known safe: ``o`` with this set holds no winning
        state, which only happens when the contract or an assumption has already failed.
        """
        hs = tuple(health_set)
        self._check_set(hs)
        hit = self.entries.get((o, hs))
        if hit is not None:
            return hit
        w = self._max_width
        if max(hi - lo + 1 for lo, hi in zip(hs[0::2], hs[1::2], strict=True)) <= w:
            return frozenset()
        # Wider than built: the intersection of the covering windows is the same intersection.
        windows = [
            [(a, min(a + w - 1, hi)) for a in range(lo, hi + 1, w)]
            for lo, hi in zip(hs[0::2], hs[1::2], strict=True)
        ]
        found = []
        for box in product(*windows):
            covered = self.entries.get((o, tuple(v for lohi in box for v in lohi)))
            if covered is not None:
                found.append(covered)
        return frozenset.intersection(*found) if found else frozenset()

    def _check_set(self, hs: tuple[int, ...]) -> None:
        if len(hs) != 2 * self._dims:
            raise ValueError(f"health set {hs!r} needs {2 * self._dims} integers")
        for lo, hi in zip(hs[0::2], hs[1::2], strict=True):
            if isinstance(lo, bool) or isinstance(hi, bool) or not (0 <= lo <= hi < N_BINS):
                raise ValueError(f"health set {hs!r} is not an interval of bins 0..{N_BINS - 1}")

    def first_allowed(self, allowed: Iterable[str]) -> str | None:
        """First command of the substitution order that is in ``allowed`` (None if empty)."""
        pool = set(allowed)
        return next((c for c in self.priority if c in pool), None)

    def model_parameters(self) -> dict[str, str]:
        """Parameters of the model this table was built from (``abstraction`` can rebuild it)."""
        return {
            k[len(MODEL_PREFIX) :]: v for k, v in self.meta.items() if k.startswith(MODEL_PREFIX)
        }

    # ------------------------------------------------------------------ serialisation

    def to_json(self) -> str:
        """Canonical text: sorted keys and entries, ASCII, no whitespace, no trailing newline."""
        rows = sorted(
            ([canonical(o), list(hs), sorted(cmds)] for (o, hs), cmds in self.entries.items()),
            key=_dump,
        )
        return _dump(
            {
                "format": FORMAT,
                "model_name": self.model_name,
                "priority": list(self.priority),
                "meta": dict(self.meta),
                "entries": rows,
            }
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> ShieldTable:
        raw = json.loads(text)
        if raw.get("format") != FORMAT:
            raise ValueError(f"not a shield table (format {raw.get('format')!r})")
        entries = {(_tuplify(o), tuple(hs)): frozenset(cmds) for o, hs, cmds in raw["entries"]}
        return cls(raw["model_name"], entries, tuple(raw["priority"]), raw["meta"])

    def save(self, path: Path) -> None:
        path.write_bytes(self.to_json().encode("ascii"))

    @classmethod
    def load(cls, path: Path, expected_sha256: str | None = None) -> ShieldTable:
        table = cls.from_json(path.read_bytes().decode("ascii"))
        if expected_sha256 is not None and table.sha256() != expected_sha256:
            raise ValueError(f"{path} hashes to {table.sha256()}, expected {expected_sha256}")
        return table


def _hvec(h: int | tuple[int, ...]) -> tuple[int, ...]:
    return (h,) if isinstance(h, int) else tuple(h)


def build_shield(
    model: AbstractModel,
    win: Winning,
    max_width: int = N_BINS,
    *,
    priority: Sequence[str] | None = None,
    provenance: Mapping[str, str] | None = None,
) -> ShieldTable:
    """Build ``A(o, B)`` for every set ``B`` of width up to ``max_width`` bins (per component).

    Raises ``ValueError`` if the initial state is not winning (the shield would be vacuous) or if
    ``max_width`` is outside 1..N_BINS. ``provenance`` (spec clause text, document, revision) is
    recorded in ``meta`` under ``spec.<key>`` and covered by the hash.
    """
    if isinstance(max_width, bool) or not 1 <= max_width <= N_BINS:
        raise ValueError(f"max_width must be in 1..{N_BINS}, got {max_width!r}")
    if model.initial() not in win.states:
        raise ValueError(f"the initial state of {model.name} is not winning")

    by_obs: dict[Hashable, dict[tuple[int, ...], frozenset[str]]] = {}
    for s in win.states:
        hv = _hvec(model.health(s))
        slot = by_obs.setdefault(model.observable(s), {})
        if hv in slot:
            raise ValueError(f"observable/health is not unique in {model.name}: {s!r}")
        slot[hv] = win.safe[s]
    dims = {len(hv) for slot in by_obs.values() for hv in slot}
    if len(dims) != 1:
        raise ValueError("health vectors have different sizes")
    (n_dims,) = dims

    intervals = [(lo, hi) for lo in range(N_BINS) for hi in range(lo, min(lo + max_width, N_BINS))]
    entries: dict[tuple[Hashable, tuple[int, ...]], frozenset[str]] = {}
    for o, slot in by_obs.items():
        for box in product(intervals, repeat=n_dims):
            members = [
                slot[h] for h in product(*(range(lo, hi + 1) for lo, hi in box)) if h in slot
            ]
            if members:
                entries[(o, tuple(v for lohi in box for v in lohi))] = frozenset.intersection(
                    *members
                )

    commands = tuple(model.commands())
    order = tuple(priority) if priority is not None else default_priority(commands)
    if sorted(order) != sorted(commands):
        raise ValueError("priority must list every command of the model exactly once")

    meta = {
        "abstraction_version": ABSTRACTION_VERSION,
        "relation_sha256": win.relation_sha256,
        "max_width": str(max_width),
        "winning_states": str(len(win.states)),
        "iterations": str(win.iterations),
    }
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        meta.update({MODEL_PREFIX + k: v for k, v in parameters().items()})
    meta.update({SPEC_PREFIX + k: v for k, v in (provenance or {}).items()})
    return ShieldTable(model.name, entries, order, meta)


def matches_model(table: ShieldTable, model: AbstractModel) -> bool:
    """True when ``table`` is exactly what ``build_shield`` produces for ``model`` today.

    The guard against hand edits and against a table that outlived a change to the abstraction: it
    re-solves the model and compares entries, priority and the digest of the relation.
    """
    win = solve(model)
    width = int(table.meta.get("max_width", "0"))
    if not 1 <= width <= N_BINS or model.initial() not in win.states:
        return False
    rebuilt = build_shield(model, win, width, priority=table.priority)
    return (
        dict(rebuilt.entries) == dict(table.entries)
        and table.meta.get("relation_sha256") == win.relation_sha256
        and table.model_name == model.name
    )
