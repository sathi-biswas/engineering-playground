"""AST and file-tree analyzer for target GitHub / local codebases."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

# Directories commonly excluded from analysis
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    ".vector_store",
    "output",
}

_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".java",
    ".rs",
    ".rb",
    ".cs",
    ".cpp",
    ".c",
    ".h",
}


@dataclass
class FunctionInfo:
    """Metadata for a single function or method."""

    name: str
    lineno: int
    end_lineno: int | None
    args: list[str] = field(default_factory=list)
    docstring: str | None = None
    is_async: bool = False


@dataclass
class ClassInfo:
    """Metadata for a class definition."""

    name: str
    lineno: int
    end_lineno: int | None
    methods: list[FunctionInfo] = field(default_factory=list)
    docstring: str | None = None
    bases: list[str] = field(default_factory=list)


@dataclass
class FileAnalysis:
    """AST-level analysis of a single source file."""

    path: str
    language: str
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    loc: int = 0
    parse_error: str | None = None


@dataclass
class RepoAnalysis:
    """Aggregate repository structure report."""

    root: str
    file_tree: list[str]
    python_files: list[FileAnalysis]
    summary: str
    file_count: int = 0
    total_loc: int = 0


class CodeParser:
    """Walk a repository, build a file tree, and parse Python ASTs."""

    def __init__(self, repo_path: str | Path, max_files: int = 500) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.max_files = max_files
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {self.repo_path}")

    def build_file_tree(self) -> list[str]:
        """Return relative paths of source files under the repo root."""
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.repo_path):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                ext = Path(name).suffix.lower()
                if ext in _CODE_EXTENSIONS or name in {"Dockerfile", "Makefile"}:
                    rel = str(Path(dirpath, name).relative_to(self.repo_path))
                    files.append(rel)
                    if len(files) >= self.max_files:
                        return sorted(files)
        return sorted(files)

    def analyze_python_file(self, rel_path: str) -> FileAnalysis:
        """Parse a Python file and extract classes, functions, and imports."""
        full = self.repo_path / rel_path
        analysis = FileAnalysis(path=rel_path, language="python")
        try:
            source = full.read_text(encoding="utf-8", errors="replace")
            analysis.loc = source.count("\n") + 1
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError as exc:
            analysis.parse_error = str(exc)
            logger.warning("Syntax error in %s: %s", rel_path, exc)
            return analysis
        except OSError as exc:
            analysis.parse_error = str(exc)
            return analysis

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    analysis.imports.append(f"{module}.{alias.name}" if module else alias.name)

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                analysis.classes.append(self._class_info(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                analysis.functions.append(self._func_info(node))

        return analysis

    def _func_info(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionInfo:
        args = [a.arg for a in node.args.args]
        return FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            args=args,
            docstring=ast.get_docstring(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )

    def _class_info(self, node: ast.ClassDef) -> ClassInfo:
        methods = [
            self._func_info(n)
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b))
            except Exception:
                bases.append(getattr(b, "id", str(b)))
        return ClassInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            methods=methods,
            docstring=ast.get_docstring(node),
            bases=bases,
        )

    def analyze_repo(self) -> RepoAnalysis:
        """Produce a full repository structure analysis."""
        tree = self.build_file_tree()
        py_files = [f for f in tree if f.endswith(".py")]
        analyses: list[FileAnalysis] = []
        total_loc = 0

        for rel in py_files[: self.max_files]:
            fa = self.analyze_python_file(rel)
            analyses.append(fa)
            total_loc += fa.loc

        summary = self._build_summary(tree, analyses, total_loc)
        logger.info(
            "Analyzed repo %s: %d files, %d Python, %d LOC",
            self.repo_path,
            len(tree),
            len(analyses),
            total_loc,
        )
        return RepoAnalysis(
            root=str(self.repo_path),
            file_tree=tree,
            python_files=analyses,
            summary=summary,
            file_count=len(tree),
            total_loc=total_loc,
        )

    def _build_summary(
        self,
        tree: list[str],
        analyses: list[FileAnalysis],
        total_loc: int,
    ) -> str:
        lines = [
            f"Repository root: `{self.repo_path}`",
            f"Total source files: {len(tree)}",
            f"Python files analyzed: {len(analyses)}",
            f"Approximate LOC: {total_loc}",
            "",
            "### Key modules",
        ]
        for fa in analyses[:40]:
            classes = ", ".join(c.name for c in fa.classes) or "—"
            funcs = ", ".join(f.name for f in fa.functions[:8]) or "—"
            lines.append(f"- `{fa.path}` (LOC {fa.loc}) classes=[{classes}] top-level=[{funcs}]")
        if len(analyses) > 40:
            lines.append(f"- … and {len(analyses) - 40} more files")
        return "\n".join(lines)

    def read_file(self, rel_path: str, max_chars: int = 30000) -> str:
        """Read a source file relative to the repo root (truncated)."""
        full = self.repo_path / rel_path
        if not full.exists():
            return f"[missing file: {rel_path}]"
        text = full.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n… [truncated at {max_chars} chars]"
        return text

    def gather_context(self, candidate_files: list[str], max_files: int = 12) -> dict[str, str]:
        """Return ``{rel_path: source}`` for the most relevant candidates."""
        context: dict[str, str] = {}
        for rel in candidate_files[:max_files]:
            # Normalize and ensure path is inside repo
            clean = rel.lstrip("./")
            if (self.repo_path / clean).exists():
                context[clean] = self.read_file(clean)
            else:
                # Fuzzy: search by basename
                basename = Path(clean).name
                for tree_file in self.build_file_tree():
                    if Path(tree_file).name == basename:
                        context[tree_file] = self.read_file(tree_file)
                        break
        return context

    def to_llm_context(self, analysis: RepoAnalysis, max_chars: int = 20000) -> str:
        """Serialize repo analysis into a compact prompt-friendly string."""
        text = analysis.summary
        if len(text) > max_chars:
            return text[:max_chars] + "\n… [truncated]"
        return text

    def file_analysis_to_dict(self, fa: FileAnalysis) -> dict[str, Any]:
        """Serialize FileAnalysis for JSON / Markdown dumps."""
        return {
            "path": fa.path,
            "language": fa.language,
            "loc": fa.loc,
            "imports": fa.imports[:30],
            "classes": [
                {
                    "name": c.name,
                    "lineno": c.lineno,
                    "methods": [m.name for m in c.methods],
                    "bases": c.bases,
                }
                for c in fa.classes
            ],
            "functions": [
                {"name": f.name, "lineno": f.lineno, "args": f.args} for f in fa.functions
            ],
            "parse_error": fa.parse_error,
        }
