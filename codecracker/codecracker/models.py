"""Shared data models for the CodeCracker pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".swift",
    ".scala",
    ".m",
    ".mm",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "out",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "coverage",
    ".next",
    ".nuxt",
    "eggs",
    "*.egg-info",
}


@dataclass
class ImportEdge:
    """A directed import/call relationship between modules."""

    source: str
    target: str
    kind: str = "import"  # import | from_import | require | call


@dataclass
class FileFacts:
    """AST / heuristic facts extracted from a single source file."""

    path: str
    language: str
    lines: int
    imports: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    docstring: str | None = None
    exports: list[str] = field(default_factory=list)
    complexity_hint: int = 0  # rough: defs + classes + branches


@dataclass
class RepoMap:
    """Structural map of a cloned repository."""

    repo_url: str
    repo_name: str
    root: Path
    tree_text: str
    files: list[FileFacts] = field(default_factory=list)
    edges: list[ImportEdge] = field(default_factory=list)
    language_counts: dict[str, int] = field(default_factory=dict)
    entrypoints: list[str] = field(default_factory=list)
    key_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuidedStep:
    """One stop on the file-by-file guided tour."""

    order: int
    path: str
    title: str
    why: str
    what_to_look_for: list[str]
    next_hint: str | None = None


@dataclass
class ArchitectureReport:
    """LLM (or heuristic) synthesis ready for README generation."""

    overview: str
    architecture_summary: str
    data_flow: str
    mermaid: str
    key_files: list[dict[str, str]]
    guided_tour: list[GuidedStep]
    how_to_extend: str
    raw_prompt: str = ""
    provider: str = "heuristic"
