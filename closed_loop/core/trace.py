"""``Trace``: the dense per-quarter record of one episode, read by ``spec`` and ``eval``.

Interface text: docs/adr/0002-interfaces.md, section 3.2.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.types import N_BINS, Command, Op, Permit

# "Large when never": minutes since the last gas test pass when none has passed, and the permit
# margin when no permit is open. Both stay finite so that float32 arithmetic and min() are safe.
GAS_AGE_NEVER_MIN = 1.0e6
PERMIT_MARGIN_NONE_MIN = 1.0e6

_OP_INDEX = {op: i for i, op in enumerate(Op)}
_PERMIT_INDEX = {p: i for i, p in enumerate(Permit)}

# Per-asset fields, with the dtype each array must have.
_PER_ASSET_DTYPES: dict[str, type[np.generic]] = {
    "op": np.int8,
    "permit": np.int8,
    "occupied": np.bool_,
    "gas_age_min": np.float32,
    "alarm": np.bool_,
    "iso_proven": np.bool_,
    "permit_margin_min": np.float32,
    "true_rul_days": np.float32,
    "health_bin": np.int8,
    "set_lo": np.int8,
    "set_hi": np.int8,
    "uncontrolled_failure": np.bool_,
}


def _same_bits(a: NDArray, b: NDArray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def op_index(op: Op) -> int:
    """Position of ``op`` in ``list(Op)``, the encoding of ``Trace.op``."""
    return _OP_INDEX[op]


def permit_index(permit: Permit) -> int:
    """Position of ``permit`` in ``list(Permit)``, the encoding of ``Trace.permit``."""
    return _PERMIT_INDEX[permit]


@dataclass(eq=False)
class Trace:
    """Dense per-quarter record of one episode (numpy arrays of length ``n_q``).

    Mutable on purpose: the plant fills the arrays in place as the episode runs. Construction
    checks keys, lengths and dtypes, and ``validate`` repeats the check after in-place edits.
    Use ``Trace.empty`` to get correctly typed arrays. Equality is identity, because a numpy
    comparison inside ``==`` is ambiguous; compare two traces with ``identical_to``.
    """

    n_q: int
    asset_ids: tuple[str, ...]
    op: dict[str, NDArray[np.int8]] = field(default_factory=dict)  # index into list(Op)
    permit: dict[str, NDArray[np.int8]] = field(default_factory=dict)  # index into list(Permit)
    occupied: dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    gas_age_min: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    alarm: dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    iso_proven: dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    permit_margin_min: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    true_rul_days: dict[str, NDArray[np.float32]] = field(default_factory=dict)  # sim-only truth
    health_bin: dict[str, NDArray[np.int8]] = field(default_factory=dict)  # sim-only
    set_lo: dict[str, NDArray[np.int8]] = field(default_factory=dict)  # sets given to the shield
    set_hi: dict[str, NDArray[np.int8]] = field(default_factory=dict)
    throughput: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, np.float32))
    uncontrolled_failure: dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    # (quarter, command, executed, overridden_by_shield)
    commands: list[tuple[int, Command, bool, bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def empty(cls, n_q: int, asset_ids: tuple[str, ...]) -> Self:
        """A trace of ``n_q`` quarters with every array allocated and neutral.

        Neutral means: ``RUN`` with no permit, nobody inside, no alarm, nothing proven, gas test
        and permit margin at their "large" values, truth unrecorded (NaN RUL, bin 0) and the
        perception set at its widest, ``(0, N_BINS - 1)``, which claims nothing.
        """
        ids = tuple(asset_ids)

        def per_asset(dtype: type[np.generic], fill: float | bool) -> dict[str, NDArray]:
            return {a: np.full(n_q, fill, dtype=dtype) for a in ids}

        return cls(
            n_q=n_q,
            asset_ids=ids,
            op=per_asset(np.int8, op_index(Op.RUN)),
            permit=per_asset(np.int8, permit_index(Permit.NONE)),
            occupied=per_asset(np.bool_, False),
            gas_age_min=per_asset(np.float32, GAS_AGE_NEVER_MIN),
            alarm=per_asset(np.bool_, False),
            iso_proven=per_asset(np.bool_, False),
            permit_margin_min=per_asset(np.float32, PERMIT_MARGIN_NONE_MIN),
            true_rul_days=per_asset(np.float32, np.nan),
            health_bin=per_asset(np.int8, 0),
            set_lo=per_asset(np.int8, 0),
            set_hi=per_asset(np.int8, N_BINS - 1),
            throughput=np.zeros(n_q, dtype=np.float32),
            uncontrolled_failure=per_asset(np.bool_, False),
        )

    def validate(self) -> None:
        """Raise ``ValueError`` on a wrong key set or length and ``TypeError`` on a wrong dtype."""
        if isinstance(self.n_q, bool) or not isinstance(self.n_q, int | np.integer) or self.n_q < 0:
            raise ValueError(f"n_q must be a non-negative int, got {self.n_q!r}")
        if not isinstance(self.asset_ids, tuple) or not all(
            isinstance(a, str) for a in self.asset_ids
        ):
            raise ValueError("asset_ids must be a tuple of str")
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError(f"asset_ids has duplicates: {self.asset_ids}")
        expected = set(self.asset_ids)
        for name, dtype in _PER_ASSET_DTYPES.items():
            by_asset = getattr(self, name)
            if set(by_asset) != expected:
                missing = sorted(expected - set(by_asset))
                extra = sorted(set(by_asset) - expected)
                raise ValueError(f"Trace.{name}: missing assets {missing}, unexpected {extra}")
            for asset, arr in by_asset.items():
                self._check_array(f"{name}[{asset!r}]", arr, dtype)
        self._check_array("throughput", self.throughput, np.float32)
        for item in self.commands:
            if len(item) != 4 or not isinstance(item[1], Command):
                raise ValueError(
                    "commands must hold (quarter, Command, executed, overridden) tuples"
                )

    def _check_array(self, label: str, arr: object, dtype: type[np.generic]) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Trace.{label}: expected a numpy array, got {type(arr).__name__}")
        if arr.dtype != np.dtype(dtype):
            raise TypeError(f"Trace.{label}: dtype {arr.dtype} but {np.dtype(dtype)} is required")
        if arr.shape != (self.n_q,):
            raise ValueError(f"Trace.{label}: shape {arr.shape} but ({self.n_q},) is required")

    def identical_to(self, other: "Trace") -> bool:
        """Bit-for-bit equality of every array and of the command log (NaN equals NaN).

        Bit-for-bit means ``-0.0`` differs from ``0.0``, which is what the twin tests want.
        """
        if self.n_q != other.n_q or self.asset_ids != other.asset_ids:
            return False
        for name in _PER_ASSET_DTYPES:
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine.keys() != theirs.keys():
                return False
            if any(not _same_bits(mine[a], theirs[a]) for a in mine):
                return False
        if not _same_bits(self.throughput, other.throughput):
            return False
        return self.commands == other.commands
