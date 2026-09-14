"""Tests for Stage 1b — AST / heuristic parsing & code-file filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from codecracker.ast_parser import (
    iter_code_files,
    parse_file,
    parse_repository,
)
from codecracker.config import Settings


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Minimal multi-language tree with noise dirs that must be skipped."""
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    src = tmp_path / "src" / "demo"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text('"""Demo package."""\n', encoding="utf-8")
    (src / "app.py").write_text(
        '''\
"""Application entry."""
import os
from pathlib import Path

class App:
    def run(self) -> None:
        if True:
            pass

def main() -> None:
    App().run()
''',
        encoding="utf-8",
    )
    (src / "util.js").write_text(
        "import fs from 'fs';\nconst x = require('path');\nexport function helper() {}\n",
        encoding="utf-8",
    )
    (src / "service.go").write_text(
        'package main\n\nimport (\n\t"fmt"\n\t"net/http"\n)\n\nfunc Serve() {}\n',
        encoding="utf-8",
    )

    # Noise: should be ignored by iter_code_files
    nm = tmp_path / "node_modules" / "left-pad"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = 1;\n", encoding="utf-8")
    junk = tmp_path / "build"
    junk.mkdir()
    (junk / "out.py").write_text("print('nope')\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not code\n", encoding="utf-8")

    return tmp_path


class TestIterCodeFiles:
    def test_includes_code_extensions(self, sample_repo: Path) -> None:
        settings = Settings(max_files=100)
        paths = iter_code_files(sample_repo, settings)
        rels = {str(p.relative_to(sample_repo)).replace("\\", "/") for p in paths}
        assert "src/demo/app.py" in rels
        assert "src/demo/util.js" in rels
        assert "src/demo/service.go" in rels

    def test_skips_vendor_and_non_code(self, sample_repo: Path) -> None:
        settings = Settings(max_files=100)
        paths = iter_code_files(sample_repo, settings)
        rels = {str(p.relative_to(sample_repo)).replace("\\", "/") for p in paths}
        assert "node_modules/left-pad/index.js" not in rels
        assert "build/out.py" not in rels
        assert "notes.txt" not in rels
        assert "README.md" not in rels

    def test_respects_max_files(self, sample_repo: Path) -> None:
        settings = Settings(max_files=2)
        paths = iter_code_files(sample_repo, settings)
        assert len(paths) == 2


class TestParseFile:
    def test_python_ast_captures_imports_classes_functions(self, sample_repo: Path) -> None:
        path = sample_repo / "src" / "demo" / "app.py"
        facts, edges = parse_file(sample_repo, path)
        assert facts.language == "python"
        assert "os" in facts.imports
        assert "pathlib" in facts.imports or any("pathlib" in i for i in facts.imports)
        assert "App" in facts.classes
        assert "main" in facts.functions
        assert facts.docstring == "Application entry."
        assert facts.complexity_hint >= 3
        assert any(e.source.endswith("app.py") and e.target == "os" for e in edges)

    def test_python_syntax_error_still_returns_facts(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.py"
        bad.write_text("def (\n", encoding="utf-8")
        facts, edges = parse_file(tmp_path, bad)
        assert facts.language == "python"
        assert facts.path == "broken.py"
        assert edges == []

    def test_javascript_heuristic_imports(self, sample_repo: Path) -> None:
        path = sample_repo / "src" / "demo" / "util.js"
        facts, edges = parse_file(sample_repo, path)
        assert facts.language == "javascript"
        assert "fs" in facts.imports
        assert "path" in facts.imports
        assert "helper" in facts.functions
        assert len(edges) >= 2

    def test_go_heuristic_imports(self, sample_repo: Path) -> None:
        path = sample_repo / "src" / "demo" / "service.go"
        facts, edges = parse_file(sample_repo, path)
        assert facts.language == "go"
        assert "fmt" in facts.imports
        assert "net/http" in facts.imports
        assert "Serve" in facts.functions


class TestParseRepository:
    def test_parse_repository_aggregates(self, sample_repo: Path) -> None:
        settings = Settings(max_files=50)
        files, edges, langs = parse_repository(sample_repo, settings)
        assert len(files) >= 3
        assert langs.get("python", 0) >= 2
        assert "javascript" in langs or "go" in langs
        assert len(edges) >= 1
        # node_modules / build must not appear
        paths = {f.path for f in files}
        assert not any(p.startswith("node_modules/") for p in paths)
        assert not any(p.startswith("build/") for p in paths)
