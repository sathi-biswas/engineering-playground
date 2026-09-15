"""Tests for tiktoken TokenBudgetingEngine in context_builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from codecracker.config import Settings
from codecracker.context_builder import (
    TokenBudgetingEngine,
    build_llm_payload,
    build_repo_map,
    pack_llm_context,
)
from codecracker.models import FileFacts, ImportEdge, RepoMap


def _fat_repo_map() -> RepoMap:
    files = []
    edges = []
    for i in range(40):
        path = f"src/pkg/module_{i}.py"
        files.append(
            FileFacts(
                path=path,
                language="python",
                lines=120 + i,
                imports=[f"pkg.dep_{j}" for j in range(20)],
                classes=[f"Class{i}A", f"Class{i}B", f"Class{i}C"],
                functions=[f"fn_{i}_{j}" for j in range(15)],
                docstring="A" * 300,
                complexity_hint=20,
            )
        )
        for j in range(10):
            edges.append(ImportEdge(source=path, target=f"pkg.dep_{j}"))
    tree_lines = ["repo/"] + [f"├── module_{i}.py" for i in range(200)]
    return RepoMap(
        repo_url="https://github.com/example/fat",
        repo_name="example/fat",
        root=Path("/tmp/fat"),
        tree_text="\n".join(tree_lines),
        files=files,
        edges=edges,
        language_counts={"python": 40},
        entrypoints=["src/pkg/module_0.py"],
        key_files=[f.path for f in files[:25]],
        metadata={},
    )


class TestTokenBudgetingEngine:
    def test_count_tokens_positive(self) -> None:
        engine = TokenBudgetingEngine(max_tokens=1000)
        assert engine.count_tokens("hello world") > 0

    def test_fit_payload_respects_budget(self) -> None:
        engine = TokenBudgetingEngine(max_tokens=400, encoding_name="cl100k_base")
        payload = build_llm_payload(_fat_repo_map())
        raw = engine.serialize(payload)
        assert engine.count_tokens(raw) > 400  # would overflow without trim

        report = engine.fit_payload(payload)
        assert report.token_count <= report.token_budget
        assert report.trimmed is True
        assert report.trim_steps  # at least one shrink step

    def test_small_payload_not_trimmed(self) -> None:
        engine = TokenBudgetingEngine(max_tokens=50_000)
        payload = {
            "repo_name": "tiny",
            "languages": {"python": 1},
            "entrypoints": ["main.py"],
            "folder_tree": "tiny/\n└── main.py\n",
            "key_files": [{"path": "main.py", "functions": ["main"]}],
            "import_graph_top": [],
            "file_count": 1,
        }
        report = engine.fit_payload(payload)
        assert report.trimmed is False
        assert report.token_count <= report.token_budget

    def test_output_reserve_reduces_usable_budget(self) -> None:
        engine = TokenBudgetingEngine(max_tokens=1000, output_reserve=400)
        assert engine.max_tokens == 600

    def test_pack_llm_context_records_metadata(self, mapped_repo: Path) -> None:
        settings = Settings(max_files=50, llm_context_token_budget=800)
        repo_map = build_repo_map(
            "https://github.com/example/sample",
            mapped_repo,
            settings,
        )
        ctx = pack_llm_context(repo_map, settings=settings)
        assert ctx
        budget = repo_map.metadata.get("token_budget")
        assert budget is not None
        assert budget["token_count"] <= budget["token_budget"]
        assert "encoding_name" in budget

    def test_pack_under_tight_budget(self) -> None:
        settings = Settings(llm_context_token_budget=350)
        repo_map = _fat_repo_map()
        ctx = pack_llm_context(repo_map, settings=settings)
        engine = TokenBudgetingEngine(max_tokens=350)
        assert engine.count_tokens(ctx) <= 350
        assert repo_map.metadata["token_budget"]["trimmed"] is True


@pytest.fixture
def mapped_repo(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# Sample\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sample'\n", encoding="utf-8")
    pkg = tmp_path / "src" / "sample"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""Sample package."""\n', encoding="utf-8")
    (pkg / "app.py").write_text(
        "def main() -> None:\n    pass\n\nclass Service:\n    pass\n",
        encoding="utf-8",
    )
    return tmp_path
