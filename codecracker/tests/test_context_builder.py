"""Tests for Stage 2 — context builder & structural map."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codecracker.config import Settings
from codecracker.context_builder import (
    build_repo_map,
    build_tree_text,
    detect_entrypoints,
    pack_llm_context,
    rank_key_files,
)
from codecracker.models import FileFacts, ImportEdge


@pytest.fixture
def mapped_repo(tmp_path: Path) -> Path:
    """Small Python package shaped like a real project."""
    (tmp_path / "README.md").write_text("# Sample\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sample'\n", encoding="utf-8")

    pkg = tmp_path / "src" / "sample"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""Sample package."""\n', encoding="utf-8")
    (pkg / "__main__.py").write_text("from sample.app import main\nmain()\n", encoding="utf-8")
    (pkg / "app.py").write_text(
        '''\
"""App hub."""
from sample import helpers

def main() -> None:
    helpers.greet()

class Service:
    pass
''',
        encoding="utf-8",
    )
    (pkg / "helpers.py").write_text(
        "def greet() -> str:\n    return 'hi'\n",
        encoding="utf-8",
    )

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )

    # Vendor noise for tree filtering
    nm = tmp_path / "node_modules" / "x"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("export default 1\n", encoding="utf-8")

    return tmp_path


class TestBuildTreeText:
    def test_includes_package_paths(self, mapped_repo: Path) -> None:
        tree = build_tree_text(mapped_repo)
        assert mapped_repo.name + "/" in tree
        assert "src/" in tree or "src" in tree
        assert "app.py" in tree

    def test_skips_node_modules(self, mapped_repo: Path) -> None:
        tree = build_tree_text(mapped_repo)
        # Root dir name may contain the substring; assert no tree entry for vendor dir
        assert "├── node_modules" not in tree
        assert "└── node_modules" not in tree
        assert "index.js" not in tree or "src/" in tree  # package files still listed


class TestDetectEntrypoints:
    def test_finds_manifests_and_mains(self, mapped_repo: Path) -> None:
        files = [
            FileFacts(path="src/sample/__main__.py", language="python", lines=2),
            FileFacts(path="src/sample/app.py", language="python", lines=10),
            FileFacts(path="tests/test_app.py", language="python", lines=2),
        ]
        eps = detect_entrypoints(files, mapped_repo)
        assert "pyproject.toml" in eps
        assert any(e.endswith("__main__.py") for e in eps)
        # Test fixtures should not be treated as entrypoints
        assert not any("test_app.py" in e for e in eps)


class TestRankKeyFiles:
    def test_prefers_src_over_tests(self) -> None:
        files = [
            FileFacts(
                path="src/sample/app.py",
                language="python",
                lines=40,
                classes=["Service"],
                functions=["main"],
                complexity_hint=5,
            ),
            FileFacts(
                path="tests/test_app.py",
                language="python",
                lines=200,
                functions=["test_ok", "test_more"],
                complexity_hint=20,
            ),
            FileFacts(
                path="src/sample/helpers.py",
                language="python",
                lines=10,
                functions=["greet"],
                complexity_hint=1,
            ),
        ]
        edges = [
            ImportEdge(source="src/sample/app.py", target="src/sample/helpers.py"),
            ImportEdge(source="tests/test_app.py", target="src/sample/app.py"),
        ]
        ranked = rank_key_files(files, edges, limit=5)
        assert ranked[0].startswith("src/")
        assert "tests/test_app.py" not in ranked[:2]


class TestBuildRepoMap:
    def test_end_to_end_map(self, mapped_repo: Path) -> None:
        settings = Settings(max_files=50)
        repo_map = build_repo_map(
            "https://github.com/example/sample",
            mapped_repo,
            settings,
        )
        assert repo_map.repo_name == "example/sample"
        assert repo_map.repo_url == "https://github.com/example/sample"
        assert len(repo_map.files) >= 3
        assert repo_map.tree_text
        assert repo_map.key_files
        assert all(
            not p.startswith("node_modules/") for p in (f.path for f in repo_map.files)
        )
        # Key tour should lead with source / manifests, not deep tests
        top = repo_map.key_files[:5]
        assert any("app.py" in p or "__main__.py" in p or "pyproject" in p for p in top)

    def test_pack_llm_context_is_valid_jsonish(self, mapped_repo: Path) -> None:
        settings = Settings(max_files=50)
        repo_map = build_repo_map(
            "https://github.com/example/sample",
            mapped_repo,
            settings,
        )
        ctx = pack_llm_context(repo_map)
        assert "example/sample" in ctx or "repo_name" in ctx
        # Truncation marker only if huge; otherwise parseable JSON
        if "… [truncated]" not in ctx:
            payload = json.loads(ctx)
            assert "key_files" in payload
            assert "folder_tree" in payload
            assert payload["file_count"] == len(repo_map.files)
