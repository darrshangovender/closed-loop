"""The sibling lock records what was tested: installed versions and tree hashes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "integration" / "siblings.lock"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "hash_siblings", ROOT / "scripts/hash_siblings.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTreeHash:
    def test_line_endings_do_not_change_the_hash(self, tool, tmp_path: Path) -> None:
        a, b = tmp_path / "a" / "pkg", tmp_path / "b" / "pkg"
        for root, text in ((a, b"x = 1\ny = 2\n"), (b, b"x = 1\r\ny = 2\r\n")):
            root.mkdir(parents=True)
            (root / "m.py").write_bytes(text)
        assert tool.tree_sha256(a) == tool.tree_sha256(b)

    def test_content_name_and_extra_files_change_it(self, tool, tmp_path: Path) -> None:
        root = tmp_path / "pkg"
        root.mkdir()
        (root / "m.py").write_text("x = 1\n")
        base = tool.tree_sha256(root)
        (root / "m.py").write_text("x = 2\n")
        assert tool.tree_sha256(root) != base
        (root / "m.py").write_text("x = 1\n")
        assert tool.tree_sha256(root) == base
        (root / "n.py").write_text("")
        assert tool.tree_sha256(root) != base

    def test_caches_are_ignored(self, tool, tmp_path: Path) -> None:
        root = tmp_path / "pkg"
        (root / "__pycache__").mkdir(parents=True)
        (root / "m.py").write_text("x = 1\n")
        base = tool.tree_sha256(root)
        (root / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"\x00\x01")
        (root / "m.pyc").write_bytes(b"\x00")
        assert tool.tree_sha256(root) == base

    def test_the_manifest_is_sorted_by_path(self, tool, tmp_path: Path) -> None:
        root = tmp_path / "pkg"
        root.mkdir()
        for name in ("b.py", "a.py", "c.py"):
            (root / name).write_text(name)
        paths = [line.split("  ", 1)[1] for line in tool.manifest(root)]
        assert paths == ["pkg/a.py", "pkg/b.py", "pkg/c.py"]


class TestLockFile:
    def test_it_names_the_three_siblings_by_direct_url_only(self) -> None:
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        assert set(lock["siblings"]) == {"asset-health", "maintenance-agent", "plant-copilot"}
        for entry in lock["siblings"].values():
            assert entry["source"].startswith("git+https://")
        assert "hash_method" in lock

    @pytest.mark.f2
    def test_the_installed_siblings_are_the_ones_that_were_tested(self, tool) -> None:
        lock = json.loads(LOCK.read_text(encoding="utf-8"))["siblings"]
        installed = tool.describe()
        for key, info in installed.items():
            assert (info["version"], info["files"], info["tree_sha256"]) == (
                lock[key]["version"],
                lock[key]["files"],
                lock[key]["tree_sha256"],
            ), f"{key} differs from integration/siblings.lock: run scripts/hash_siblings.py --write"
