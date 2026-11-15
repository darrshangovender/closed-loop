"""Keyed random streams. All randomness in the repository comes from ``rng_for``.

Interface text: docs/adr/0002-interfaces.md, sections 1 (rule 5) and 3.2.
"""

import operator
import zlib

import numpy as np


def _crc(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def _key_part(part: int | str, what: str = "stream id") -> int:
    if isinstance(part, str):
        return _crc(part)
    if isinstance(part, bool):
        raise TypeError(f"{what} must be int or str, not bool")
    try:
        value = operator.index(part)  # accepts numpy integers, refuses floats
    except TypeError:
        raise TypeError(f"{what} must be int or str, not {type(part).__name__}") from None
    if value < 0:
        raise ValueError(f"{what} must be non-negative, got {value}")
    return value


def rng_for(root_seed: int, stream: str, *ids: int | str) -> np.random.Generator:
    """Keyed generator: PCG64(SeedSequence(root_seed, spawn_key=(crc32(stream), *int_ids))).

    ``str`` ids are CRC-32'd (UTF-8, unsigned). Same arguments give the same stream regardless of
    any other draw, in this process or another, so a draw for one asset never shifts a draw for
    another. The number of ids is part of the key: ``(s, 1)`` and ``(s, 1, 0)`` are different
    streams. Callers must not rely on a string id and an int id being distinct when the int equals
    the string's CRC-32.
    """
    if not isinstance(stream, str):
        raise TypeError(f"stream must be str, not {type(stream).__name__}")
    if isinstance(root_seed, str):
        raise TypeError("root_seed must be an int")
    seed = _key_part(root_seed, "root_seed")
    spawn_key = (_crc(stream), *(_key_part(i) for i in ids))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=spawn_key)))
