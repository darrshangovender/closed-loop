import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from closed_loop.core.rng import rng_for

REPO = Path(__file__).resolve().parents[2]
N_DRAWS = 10_000
# Computed once from numpy's own PCG64 and SeedSequence, outside closed_loop.
GOLDEN_WORDS = [1452550723029609731, 14370371613709577061, 9866815405857983754]


def draws(rng, n=N_DRAWS):
    return rng.random(n)


class TestKeyedStreams:
    def test_same_arguments_same_stream(self):
        a = draws(rng_for(7, "degradation", 3, "pump-a"))
        b = draws(rng_for(7, "degradation", 3, "pump-a"))
        assert np.array_equal(a, b)

    def test_matches_the_documented_construction(self):
        """PCG64(SeedSequence(root, spawn_key=(crc32(stream), *int_ids))), str ids crc32'd."""
        key = (zlib.crc32(b"alpha"), 7, zlib.crc32("pump-é".encode()))
        reference = np.random.Generator(np.random.PCG64(np.random.SeedSequence(123, spawn_key=key)))
        mine = rng_for(123, "alpha", 7, "pump-é")
        assert np.array_equal(reference.random(50), mine.random(50))

    def test_golden_first_words(self):
        """Pins the stream itself: a change of construction or numpy's PCG64 turns this red."""
        rng = rng_for(2026, "golden", 1, "x")
        assert [int(rng.bit_generator.random_raw()) for _ in range(3)] == GOLDEN_WORDS

    @pytest.mark.parametrize(
        "other",
        [
            (8, "degradation", (3, "pump-a")),  # root seed
            (7, "observation", (3, "pump-a")),  # stream
            (7, "degradation", (4, "pump-a")),  # int id
            (7, "degradation", (3, "pump-b")),  # str id
            (7, "degradation", (3,)),  # fewer ids
            (7, "degradation", (3, "pump-a", 0)),  # more ids
            (7, "degradation", ("pump-a", 3)),  # order of ids
        ],
    )
    def test_different_key_gives_an_independent_stream(self, other):
        root, stream, ids = other
        a = draws(rng_for(7, "degradation", 3, "pump-a"))
        b = draws(rng_for(root, stream, *ids))
        assert not np.array_equal(a, b)
        r = np.corrcoef(a, b)[0, 1]
        # 4 standard errors of a null correlation at n = 10^4; the seeds are fixed, so this is
        # a deterministic check and cannot flake
        assert abs(r) < 4 / np.sqrt(N_DRAWS)

    def test_sibling_ids_are_pairwise_uncorrelated(self):
        streams = np.array([draws(rng_for(1, "life", i)) for i in range(12)])
        corr = np.corrcoef(streams)
        off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
        assert np.abs(off_diagonal).max() < 5 / np.sqrt(N_DRAWS)

    def test_drawing_from_one_stream_does_not_change_another(self):
        alone = draws(rng_for(5, "b", 1), 100)
        a = rng_for(5, "a", 1)
        b = rng_for(5, "b", 1)
        draws(a, 5_000)  # heavy use of A first
        interleaved = draws(b, 100)
        assert np.array_equal(alone, interleaved)

    def test_creation_order_does_not_matter(self):
        first = rng_for(5, "x", 1).random(10)
        _ = [rng_for(5, "y", i) for i in range(20)]
        again = rng_for(5, "x", 1).random(10)
        assert np.array_equal(first, again)

    def test_state_is_per_call(self):
        rng = rng_for(5, "x")
        rng.random(100)
        assert np.array_equal(rng_for(5, "x").random(3), rng_for(5, "x").random(3))

    def test_numpy_integer_ids_equal_python_ints(self):
        a = rng_for(np.int64(9), "x", np.int32(4)).random(5)
        b = rng_for(9, "x", 4).random(5)
        assert np.array_equal(a, b)

    def test_unicode_stream_and_ids(self):
        # Arabic ids are plain UTF-8 text here; the test pins that they are accepted and stable
        a = rng_for(1, "ملاحظة", "مضخة").random(3)
        b = rng_for(1, "ملاحظة", "مضخة").random(3)
        assert np.array_equal(a, b)

    def test_returns_a_pcg64_generator(self):
        rng = rng_for(0, "x")
        assert isinstance(rng, np.random.Generator)
        assert isinstance(rng.bit_generator, np.random.PCG64)

    def test_huge_root_seed_is_accepted(self):
        assert rng_for(2**100, "x").random() != rng_for(2**100 + 1, "x").random()


class TestRefusals:
    # numpy's own SeedSequence also says "non-negative", so match our message, not just the word
    def test_negative_root_seed(self):
        with pytest.raises(ValueError, match="root_seed must be non-negative, got -1"):
            rng_for(-1, "x")

    def test_negative_id(self):
        with pytest.raises(ValueError, match="stream id must be non-negative, got -3"):
            rng_for(1, "x", -3)

    @pytest.mark.parametrize("bad", [1.5, None, b"x", (1,), True])
    def test_bad_id_type(self, bad):
        with pytest.raises(TypeError):
            rng_for(1, "x", bad)

    def test_str_root_seed(self):
        with pytest.raises(TypeError):
            rng_for("7", "x")  # type: ignore[arg-type]

    def test_non_str_stream(self):
        with pytest.raises(TypeError):
            rng_for(1, 5)  # type: ignore[arg-type]


CHILD = """
import json, sys
from closed_loop.core.rng import rng_for
rng = rng_for(2026, "cross-process", 4, "pump-a")
print(json.dumps([int(rng.bit_generator.random_raw()) for _ in range(4)] + rng.random(4).tolist()))
"""


def test_stable_across_processes():
    in_process = rng_for(2026, "cross-process", 4, "pump-a")
    expected = [int(in_process.bit_generator.random_raw()) for _ in range(4)]
    expected += in_process.random(4).tolist()
    # PYTHONHASHSEED differs per process: any use of hash() would show up as a difference
    for hash_seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", CHILD],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
        assert json.loads(out) == expected
