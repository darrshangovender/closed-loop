"""P10: the pre-registration hash and the code hash."""

import hashlib
from pathlib import Path

import pytest

from closed_loop.audit.prereg import (
    code_sha256,
    file_sha256,
    normalise_bytes,
    normalised_bytes,
    prereg_sha256,
)
from closed_loop.audit.seeds import heldout_seed

TEXT = "# Pre-registration\n\nClaim C1: miss rate below 0.01.\nClaim C2: shield is sound.\n"


def write(path: Path, data: str | bytes) -> Path:
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


class TestNormalisation:
    def test_crlf_and_lf_copies_give_the_same_hash(self, tmp_path: Path) -> None:
        lf = write(tmp_path / "lf.md", TEXT)
        crlf = write(tmp_path / "crlf.md", TEXT.replace("\n", "\r\n"))
        assert prereg_sha256(lf) == prereg_sha256(crlf)
        assert normalised_bytes(crlf) == TEXT.encode()

    def test_mixed_line_endings(self, tmp_path: Path) -> None:
        mixed = write(tmp_path / "m.md", "a\r\nb\nc\r\n")
        assert normalised_bytes(mixed) == b"a\nb\nc\n"

    def test_trailing_spaces_and_tabs_are_ignored(self, tmp_path: Path) -> None:
        clean = write(tmp_path / "clean.md", TEXT)
        spaces = write(tmp_path / "spaces.md", TEXT.replace("\n", "   \n"))
        mixed_crlf = write(tmp_path / "mixed.md", TEXT.replace("\n", " \t \r\n"))
        assert prereg_sha256(spaces) == prereg_sha256(mixed_crlf) == prereg_sha256(clean)

    def test_the_last_line_without_a_newline_is_stripped_too(self) -> None:
        assert normalise_bytes(b"abc   ") == b"abc"

    def test_leading_and_inner_whitespace_matter(self) -> None:
        assert normalise_bytes(b"  a  b\n") == b"  a  b\n"
        assert normalise_bytes(b"a\n\n\nb") == b"a\n\n\nb"  # blank lines are content

    def test_a_missing_final_newline_is_a_difference(self, tmp_path: Path) -> None:
        with_nl = write(tmp_path / "a.md", TEXT)
        without = write(tmp_path / "b.md", TEXT.rstrip("\n"))
        assert prereg_sha256(with_nl) != prereg_sha256(without)

    def test_a_lone_carriage_return_is_not_a_line_ending(self) -> None:
        assert normalise_bytes(b"a\rb\n") == b"a\rb\n"

    def test_bom_is_kept(self, tmp_path: Path) -> None:
        plain = write(tmp_path / "a.md", TEXT)
        bom = write(tmp_path / "b.md", b"\xef\xbb\xbf" + TEXT.encode())
        assert prereg_sha256(plain) != prereg_sha256(bom)

    def test_non_ascii_text_survives(self, tmp_path: Path) -> None:
        path = write(tmp_path / "ar.md", "مرحبا\r\n")
        assert normalised_bytes(path) == "مرحبا\n".encode()


class TestHash:
    def test_is_the_sha256_of_the_normalised_bytes(self, tmp_path: Path) -> None:
        path = write(tmp_path / "x.md", TEXT.replace("\n", "\r\n"))
        assert prereg_sha256(path) == hashlib.sha256(TEXT.encode()).hexdigest()
        assert file_sha256(path) == prereg_sha256(path)

    def test_one_changed_character_changes_the_hash_and_every_heldout_seed(
        self, tmp_path: Path
    ) -> None:
        a = write(tmp_path / "a.md", TEXT)
        b = write(tmp_path / "b.md", TEXT.replace("0.01", "0.02"))
        sha_a, sha_b = prereg_sha256(a), prereg_sha256(b)
        assert sha_a != sha_b
        assert all(heldout_seed(sha_a, i) != heldout_seed(sha_b, i) for i in range(50))

    def test_hash_is_lowercase_hex_of_64_characters(self, tmp_path: Path) -> None:
        sha = prereg_sha256(write(tmp_path / "a.md", ""))
        assert len(sha) == 64 and sha == sha.lower()
        assert sha == hashlib.sha256(b"").hexdigest()

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            prereg_sha256(tmp_path / "absent.md")


def make_tree(root: Path) -> Path:
    pkg = root / "pkg"
    (pkg / "sub").mkdir(parents=True)
    write(pkg / "a.py", "x = 1\n")
    write(pkg / "sub" / "b.py", "y = 2\n")
    write(pkg / "notes.txt", "not code\n")
    return pkg


class TestCodeHash:
    def test_stable_and_hex(self, tmp_path: Path) -> None:
        pkg = make_tree(tmp_path)
        first = code_sha256([pkg])
        assert first == code_sha256([pkg]) and len(first) == 64

    def test_does_not_depend_on_where_the_tree_sits(self, tmp_path: Path) -> None:
        one = make_tree(tmp_path / "one")
        two = make_tree(tmp_path / "deeper" / "two")
        assert code_sha256([one]) == code_sha256([two])

    def test_line_endings_do_not_matter(self, tmp_path: Path) -> None:
        pkg = make_tree(tmp_path / "lf")
        crlf = make_tree(tmp_path / "crlf")
        for f in crlf.rglob("*.py"):
            f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
        assert code_sha256([pkg]) == code_sha256([crlf])

    def test_a_changed_byte_a_new_file_and_a_rename_all_change_it(self, tmp_path: Path) -> None:
        pkg = make_tree(tmp_path)
        base = code_sha256([pkg])
        write(pkg / "a.py", "x = 2\n")
        edited = code_sha256([pkg])
        assert edited != base
        write(pkg / "c.py", "")
        added = code_sha256([pkg])
        assert added != edited
        (pkg / "c.py").rename(pkg / "d.py")
        assert code_sha256([pkg]) != added

    def test_swapping_the_contents_of_two_files_changes_it(self, tmp_path: Path) -> None:
        pkg = make_tree(tmp_path)
        base = code_sha256([pkg])
        write(pkg / "a.py", "y = 2\n")
        write(pkg / "sub" / "b.py", "x = 1\n")
        assert code_sha256([pkg]) != base

    def test_other_files_caches_and_environments_are_skipped(self, tmp_path: Path) -> None:
        pkg = make_tree(tmp_path)
        base = code_sha256([pkg])
        write(pkg / "notes.txt", "edited\n")
        (pkg / "__pycache__").mkdir()
        write(pkg / "__pycache__" / "a.cpython-312.py", "junk")
        (pkg / ".venv").mkdir()
        write(pkg / ".venv" / "site.py", "junk")
        assert code_sha256([pkg]) == base

    def test_suffixes_can_be_widened(self, tmp_path: Path) -> None:
        pkg = make_tree(tmp_path)
        assert code_sha256([pkg], (".py", ".txt")) != code_sha256([pkg])

    def test_a_single_file_root(self, tmp_path: Path) -> None:
        pkg = make_tree(tmp_path)
        script = write(tmp_path / "run.py", "print(1)\n")
        combined = code_sha256([pkg, script])
        assert combined != code_sha256([pkg])
        write(script, "print(2)\n")
        assert code_sha256([pkg, script]) != combined

    def test_empty_missing_and_clashing_roots_are_errors(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no .py files"):
            code_sha256([empty])
        with pytest.raises(FileNotFoundError):
            code_sha256([tmp_path / "absent"])
        a = make_tree(tmp_path / "a")
        b = make_tree(tmp_path / "b")
        with pytest.raises(ValueError, match="same path"):
            code_sha256([a, b])

    def test_the_real_package_hashes(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        assert len(code_sha256([repo / "closed_loop"])) == 64
