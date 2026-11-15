"""Held-out, F2 and published-search seeds, derived from the pre-registration hash.

Interface text: docs/adr/0002-interfaces.md, section 3.12; method: docs/adr/0001-architecture.md,
section 12. A held-out seed does not exist before the pre-registration is frozen, and any change to
the document changes every one of them.

    seed = int.from_bytes(sha256(prereg_sha || label || i)[:8], "big")

with ``prereg_sha`` as its 64 ASCII hex characters, ``label`` ``b"heldout"``, ``b"f2"`` or
``b"search"``, and ``i`` as 8 bytes, big-endian. The seed is a 64-bit unsigned integer, accepted by
``rng_for``.

The search runs that are published use ``search_seed`` (0003 A14): development had already explored
the integers 3000 to 3999, which stay for development under the role ``search-dev``.
"""

from __future__ import annotations

import hashlib
import re

__all__ = [
    "CALIB_SEEDS",
    "DEV_SEEDS",
    "SEARCH_DEV_SEEDS",
    "SEARCH_SEEDS",
    "f2_seed",
    "heldout_seed",
    "search_seed",
]

# Seed roles that are fixed in advance (0001 section 11); held-out and F2 seeds are derived.
DEV_SEEDS = range(1000, 2000)
CALIB_SEEDS = range(2000, 3000)
SEARCH_DEV_SEEDS = range(3000, 4000)  # exploring the search during the build (0003 A14)
SEARCH_SEEDS = SEARCH_DEV_SEEDS  # the name it had before the role was split

_HEX64 = re.compile(r"[0-9a-f]{64}")
_MAX_INDEX = 2**64 - 1


def _derive(prereg_sha: str, label: bytes, i: int) -> int:
    if not isinstance(prereg_sha, str) or not _HEX64.fullmatch(prereg_sha):
        raise ValueError("prereg_sha must be 64 lower-case hex characters")
    if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i <= _MAX_INDEX:
        raise ValueError(f"i must be an int in [0, 2**64), got {i!r}")
    digest = hashlib.sha256(prereg_sha.encode("ascii") + label + i.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big")


def heldout_seed(prereg_sha: str, i: int) -> int:
    """The ``i``-th held-out seed for the pre-registration with hash ``prereg_sha``."""
    return _derive(prereg_sha, b"heldout", i)


def f2_seed(prereg_sha: str, i: int) -> int:
    """The ``i``-th F2 seed: same construction, label ``b"f2"``, so the two sets are unrelated."""
    return _derive(prereg_sha, b"f2", i)


def search_seed(prereg_sha: str, i: int) -> int:
    """The ``i``-th seed of a published search run: label ``b"search"``, unrelated to the others."""
    return _derive(prereg_sha, b"search", i)
