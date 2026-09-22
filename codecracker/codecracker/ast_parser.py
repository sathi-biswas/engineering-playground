"""Stage 1b — filter non-code, parse ASTs, capture imports & call-graph edges."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.progress import track

from .config import Settings
from .models import CODE_EXTENSIONS, SKIP_DIRS, FileFacts, ImportEdge

console = Console()

EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
}

# Heuristic import extractors for non-Python languages
_JS_IMPORT = re.compile(
    r"""(?:import\s+(?:[\w*{}\s,]+\s+from\s+)?['"]([^'"]+)['"]"""
    r"""|require\(\s*['"]([^'"]+)['"]\s*\))"""
)
_GO_IMPORT = re.compile(r"""import\s+(?:\(\s*([\s\S]*?)\)|"([^"]+)")""")
_GO_IMPORT_LINE = re.compile(r'"([^"]+)"')
_JAVA_IMPORT = re.compile(r"import\s+([\w.]+)\s*;")
_RUST_USE = re.compile(r"use\s+([\w:]+)(?:\s*::\s*\{[^}]+\})?\s*;")
_RUBY_REQUIRE = re.compile(r"""(?:require|require_relative)\s+['"]([^'"]+)['"]""")


def _should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.endswith(".egg-info")


def iter_code_files(root: Path, settings: Settings) -> list[Path]:
    """Walk the tree, skipping vendor/build dirs and non-code extensions."""
    found: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(_should_skip_dir(p.name) for p in path.parents):
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > settings.max_file_bytes:
                continue
        except OSError:
            continue
        found.append(path)
    return sorted(found)[:settings.max_files]


def _rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


# ---------------------------------------------------------------------------
# Python AST
# ---------------------------------------------------------------------------


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[str] = []
        self.classes: list[str] = []
        self.functions: list[str] = []
        self.complexity = 0

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if node.level:
            mod = "." * node.level + mod
        self.imports.append(mod or ".")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.complexity += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append(node.name)
        self.complexity += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)


def _parse_python(root: Path, path: Path, text: str) -> tuple[FileFacts, list[ImportEdge]]:
    rel = _rel(root, path)
    visitor = _PythonVisitor()
    docstring = None
    try:
        tree = ast.parse(text, filename=rel)
        visitor.visit(tree)
        docstring = ast.get_docstring(tree)
    except SyntaxError:
        # Still record the file; imports stay empty
        pass

    facts = FileFacts(
        path=rel,
        language="python",
        lines=text.count("\n") + 1,
        imports=visitor.imports,
        classes=visitor.classes,
        functions=visitor.functions,
        docstring=docstring,
        complexity_hint=visitor.complexity,
    )
    edges = [
        ImportEdge(source=rel, target=imp, kind="import") for imp in visitor.imports
    ]
    return facts, edges


# ---------------------------------------------------------------------------
# Heuristic parsers (JS/TS/Go/Java/Rust/Ruby)
# ---------------------------------------------------------------------------


def _parse_js_like(root: Path, path: Path, text: str, lang: str) -> tuple[FileFacts, list[ImportEdge]]:
    rel = _rel(root, path)
    imports: list[str] = []
    for m in _JS_IMPORT.finditer(text):
        imports.append(m.group(1) or m.group(2))
    classes = re.findall(r"\b(?:export\s+)?class\s+(\w+)", text)
    functions = re.findall(
        r"\b(?:export\s+)?(?:async\s+)?function\s+(\w+)"
        r"|\b(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(",
        text,
    )
    flat_fns = [a or b for a, b in functions]
    exports = re.findall(r"export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)", text)
    facts = FileFacts(
        path=rel,
        language=lang,
        lines=text.count("\n") + 1,
        imports=imports,
        classes=classes,
        functions=flat_fns,
        exports=exports,
        complexity_hint=len(classes) + len(flat_fns),
    )
    edges = [ImportEdge(source=rel, target=i, kind="import") for i in imports]
    return facts, edges


def _parse_go(root: Path, path: Path, text: str) -> tuple[FileFacts, list[ImportEdge]]:
    rel = _rel(root, path)
    imports: list[str] = []
    for m in _GO_IMPORT.finditer(text):
        block, single = m.group(1), m.group(2)
        if single:
            imports.append(single)
        elif block:
            imports.extend(_GO_IMPORT_LINE.findall(block))
    types = re.findall(r"type\s+(\w+)\s+(?:struct|interface)", text)
    funcs = re.findall(r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", text)
    facts = FileFacts(
        path=rel,
        language="go",
        lines=text.count("\n") + 1,
        imports=imports,
        classes=types,
        functions=funcs,
        complexity_hint=len(types) + len(funcs),
    )
    edges = [ImportEdge(source=rel, target=i, kind="import") for i in imports]
    return facts, edges


def _parse_java(root: Path, path: Path, text: str) -> tuple[FileFacts, list[ImportEdge]]:
    rel = _rel(root, path)
    imports = _JAVA_IMPORT.findall(text)
    classes = re.findall(r"\b(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?class\s+(\w+)", text)
    funcs = re.findall(
        r"\b(?:public|private|protected|static|\s)+\s+[\w<>\[\]]+\s+(\w+)\s*\(",
        text,
    )
    facts = FileFacts(
        path=rel,
        language="java",
        lines=text.count("\n") + 1,
        imports=imports,
        classes=classes,
        functions=funcs[:40],
        complexity_hint=len(classes) + len(funcs),
    )
    edges = [ImportEdge(source=rel, target=i, kind="import") for i in imports]
    return facts, edges


def _parse_rust(root: Path, path: Path, text: str) -> tuple[FileFacts, list[ImportEdge]]:
    rel = _rel(root, path)
    imports = _RUST_USE.findall(text)
    structs = re.findall(r"\b(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", text)
    funcs = re.findall(r"\b(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", text)
    facts = FileFacts(
        path=rel,
        language="rust",
        lines=text.count("\n") + 1,
        imports=imports,
        classes=structs,
        functions=funcs,
        complexity_hint=len(structs) + len(funcs),
    )
    edges = [ImportEdge(source=rel, target=i, kind="import") for i in imports]
    return facts, edges


def _parse_ruby(root: Path, path: Path, text: str) -> tuple[FileFacts, list[ImportEdge]]:
    rel = _rel(root, path)
    imports = _RUBY_REQUIRE.findall(text)
    classes = re.findall(r"\bclass\s+(\w+)", text)
    funcs = re.findall(r"\bdef\s+(\w+[?!]?)", text)
    facts = FileFacts(
        path=rel,
        language="ruby",
        lines=text.count("\n") + 1,
        imports=imports,
        classes=classes,
        functions=funcs,
        complexity_hint=len(classes) + len(funcs),
    )
    edges = [ImportEdge(source=rel, target=i, kind="import") for i in imports]
    return facts, edges


def _parse_generic(root: Path, path: Path, text: str, lang: str) -> tuple[FileFacts, list[ImportEdge]]:
    rel = _rel(root, path)
    facts = FileFacts(
        path=rel,
        language=lang,
        lines=text.count("\n") + 1,
        complexity_hint=text.count("\n") // 50,
    )
    return facts, []


def parse_file(root: Path, path: Path) -> tuple[FileFacts, list[ImportEdge]]:
    """Dispatch to the right parser for a single file."""
    lang = EXT_LANG.get(path.suffix.lower(), "other")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return (
            FileFacts(path=_rel(root, path), language=lang, lines=0),
            [],
        )

    if lang == "python":
        return _parse_python(root, path, text)
    if lang in {"javascript", "typescript"}:
        return _parse_js_like(root, path, text, lang)
    if lang == "go":
        return _parse_go(root, path, text)
    if lang == "java":
        return _parse_java(root, path, text)
    if lang == "rust":
        return _parse_rust(root, path, text)
    if lang == "ruby":
        return _parse_ruby(root, path, text)
    return _parse_generic(root, path, text, lang)


def parse_repository(root: Path, settings: Settings | None = None) -> tuple[list[FileFacts], list[ImportEdge], dict[str, int]]:
    """
    Filter non-code files, parse ASTs / heuristics, and build module edges.

    Returns (file_facts, import_edges, language_counts).
    """
    settings = settings or Settings()
    paths = iter_code_files(root, settings)
    console.print(f"[cyan]⚙[/] Parsing [bold]{len(paths)}[/] code files under {root.name}")

    facts_list: list[FileFacts] = []
    edges: list[ImportEdge] = []
    lang_counter: Counter[str] = Counter()

    for path in track(paths, description="AST / heuristic parse", console=console):
        facts, file_edges = parse_file(root, path)
        facts_list.append(facts)
        edges.extend(file_edges)
        lang_counter[facts.language] += 1

    console.print(
        f"[green]✓[/] Captured {len(facts_list)} files, {len(edges)} import edges"
    )
    return facts_list, edges, dict(lang_counter)
