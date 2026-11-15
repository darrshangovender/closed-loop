"""P10: held-out and F2 seeds derived from the pre-registration hash."""

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from closed_loop.audit.seeds import (
    CALIB_SEEDS,
    DEV_SEEDS,
    SEARCH_DEV_SEEDS,
    SEARCH_SEEDS,
    f2_seed,
    heldout_seed,
    search_seed,
)
from closed_loop.core.rng import rng_for

SHA = hashlib.sha256(b"a pre-registration").hexdigest()
OTHER = hashlib.sha256(b"a pre-registration.").hexdigest()


def reference(sha: str, label: bytes, i: int) -> int:
    """The formula of 0002, section 3.12, written out again."""
    data = sha.encode("ascii") + label + i.to_bytes(8, "big")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


class TestDerivation:
    def test_matches_the_published_formula(self) -> None:
        for i in (0, 1, 2, 99, 2**40):
            assert heldout_seed(SHA, i) == reference(SHA, b"heldout", i)
            assert f2_seed(SHA, i) == reference(SHA, b"f2", i)
            assert search_seed(SHA, i) == reference(SHA, b"search", i)

    def test_golden_values_pin_the_byte_layout(self) -> None:
        """If the encoding of the hash or of i ever changes, every published seed changes."""
        sha = "0" * 64
        assert heldout_seed(sha, 0) == 6059503840759059407
        assert heldout_seed(sha, 1) == 12928460720473450230
        assert f2_seed(sha, 0) == 5076782643514194193
        assert search_seed(sha, 0) == 8691061877424323584
        assert search_seed(sha, 1) == 9329167149127230308

    def test_labels_give_unrelated_sets(self) -> None:
        held = {heldout_seed(SHA, i) for i in range(500)}
        f2 = {f2_seed(SHA, i) for i in range(500)}
        found = {search_seed(SHA, i) for i in range(500)}
        assert len(held) == len(f2) == len(found) == 500
        assert not (held & f2) and not (held & found) and not (f2 & found)

    def test_one_changed_character_in_the_document_changes_every_seed(self) -> None:
        assert all(heldout_seed(SHA, i) != heldout_seed(OTHER, i) for i in range(200))
        assert all(f2_seed(SHA, i) != f2_seed(OTHER, i) for i in range(200))
        assert all(search_seed(SHA, i) != search_seed(OTHER, i) for i in range(200))

    def test_seeds_are_64_bit_unsigned_and_do_not_collide(self) -> None:
        seeds = [heldout_seed(SHA, i) for i in range(20000)]
        assert len(set(seeds)) == len(seeds)
        assert all(0 <= s < 2**64 for s in seeds)
        assert max(seeds) > 2**63  # uses the whole range

    def test_derived_seeds_never_fall_in_the_fixed_role_ranges(self) -> None:
        roles = (*DEV_SEEDS, *CALIB_SEEDS, *SEARCH_DEV_SEEDS)
        derived = (
            {heldout_seed(SHA, i) for i in range(5000)}
            | {f2_seed(SHA, i) for i in range(5000)}
            | {search_seed(SHA, i) for i in range(5000)}
        )
        assert not derived & set(roles)

    def test_published_search_seeds_are_not_in_dev_calib_or_search_dev(self) -> None:
        # the registered counts are 10 per cell; look much further, for two documents
        for sha in (SHA, OTHER):
            found = [search_seed(sha, i) for i in range(20000)]
            for role in (DEV_SEEDS, CALIB_SEEDS, SEARCH_DEV_SEEDS):
                assert not any(role.start <= s < role.stop for s in found), role
            assert len(set(found)) == len(found)

    def test_the_search_seeds_change_when_the_preregistration_hash_changes(self) -> None:
        a = [search_seed(SHA, i) for i in range(40)]
        b = [search_seed(OTHER, i) for i in range(40)]
        assert not set(a) & set(b)

    def test_search_seeds_ask_for_a_valid_hash_and_index(self) -> None:
        with pytest.raises(ValueError, match="64 lower-case hex"):
            search_seed("nonsense", 0)
        with pytest.raises(ValueError, match="int in"):
            search_seed(SHA, -1)

    def test_role_ranges(self) -> None:
        assert (DEV_SEEDS.start, DEV_SEEDS.stop) == (1000, 2000)
        assert (CALIB_SEEDS.start, CALIB_SEEDS.stop) == (2000, 3000)
        assert (SEARCH_DEV_SEEDS.start, SEARCH_DEV_SEEDS.stop) == (3000, 4000)
        assert SEARCH_SEEDS == SEARCH_DEV_SEEDS  # the old name still means the development range

    def test_stable_across_processes(self) -> None:
        code = (
            f"from closed_loop.audit.seeds import heldout_seed\nprint(heldout_seed({SHA!r}, 7))\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONHASHSEED": "123"},
            check=False,
        )
        assert done.stdout.strip() == str(heldout_seed(SHA, 7)), done.stderr

    def test_a_derived_seed_drives_rng_for(self) -> None:
        a = rng_for(heldout_seed(SHA, 3), "plant", 1).random(4)
        b = rng_for(heldout_seed(SHA, 3), "plant", 1).random(4)
        c = rng_for(heldout_seed(SHA, 4), "plant", 1).random(4)
        assert np.array_equal(a, b) and not np.array_equal(a, c)


class TestRefusals:
    @pytest.mark.parametrize(
        "sha",
        ["", "abc", "A" * 64, "g" * 64, SHA[:-1], SHA + "0", None, 5, SHA.encode()],
    )
    def test_bad_hash(self, sha) -> None:
        with pytest.raises(ValueError, match="64 lower-case hex"):
            heldout_seed(sha, 0)
        with pytest.raises(ValueError, match="64 lower-case hex"):
            f2_seed(sha, 0)

    @pytest.mark.parametrize("i", [-1, 2**64, 1.5, "1", None, True])
    def test_bad_index(self, i) -> None:
        with pytest.raises(ValueError, match="int in"):
            heldout_seed(SHA, i)
