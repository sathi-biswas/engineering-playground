"""Stage 2 — Context Builder & Structural Map.

Builds a folder tree, ranks key files / entrypoints, and packs a compact
context blob for the LLM prompt strategy engine.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
from rich.console import Console

from .ast_parser import parse_repository
from .cloner import normalize_repo_url
from .config import Settings
from .models import FileFacts, ImportEdge, RepoMap

console = Console()

ENTRYPOINT_NAMES = {
    "main.py",
    "__main__.py",
    "app.py",
    "cli.py",
    "manage.py",
    "wsgi.py",
    "asgi.py",
    "index.js",
    "index.ts",
    "main.go",
    "main.rs",
    "Main.java",
    "server.js",
    "server.ts",
    "app.js",
    "app.ts",
}

MANIFEST_NAMES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    "README.md",
    "readme.md",
}


def build_tree_text(root: Path, max_depth: int = 4, max_entries: int = 200) -> str:
    """ASCII folder tree (code + manifests only), depth-limited."""
    lines: list[str] = [root.name + "/"]
    count = 0

    def walk(directory: Path, prefix: str, depth: int) -> None:
        nonlocal count
        if depth > max_depth or count >= max_entries:
            return
        try:
            children = sorted(
                directory.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError:
            return

        # Filter noise
        visible = []
        for child in children:
            if child.name.startswith(".") and child.name not in {".github"}:
                continue
            if child.is_dir() and child.name in {
                "node_modules",
                "vendor",
                "__pycache__",
                ".git",
                "dist",
                "build",
                "target",
                ".venv",
                "venv",
            }:
                continue
            visible.append(child)

        for i, child in enumerate(visible):
            if count >= max_entries:
                lines.append(prefix + "└── …")
                return
            last = i == len(visible) - 1
            branch = "└── " if last else "├── "
            connector = "    " if last else "│   "
            label = child.name + ("/" if child.is_dir() else "")
            lines.append(prefix + branch + label)
            count += 1
            if child.is_dir():
                walk(child, prefix + connector, depth + 1)

    walk(root, "", 1)
    return "\n".join(lines)


def _score_file(facts: FileFacts, in_degree: int, out_degree: int) -> float:
    """Higher = more central / worth reading first."""
    name = Path(facts.path).name
    path_l = facts.path.lower()
    score = 0.0
    if name in ENTRYPOINT_NAMES:
        score += 50
    if name in MANIFEST_NAMES:
        score += 30
    # Tests / fixtures / examples are useful later, not first
    if any(tok in path_l for tok in ("/test", "test_", "/spec", "/example", "/fixture")):
        score -= 40
        score -= facts.path.count("/") * 2  # deep nested fixtures sink further
    score += min(facts.complexity_hint, 40) * 0.5
    score += min(facts.lines, 500) * 0.02
    score += in_degree * 3 + out_degree * 1.5
    if facts.docstring:
        score += 5
    # Prefer shallow paths (closer to root = often more architectural)
    depth = facts.path.count("/")
    score += max(0, 8 - depth)
    # Prefer package source roots (src/, lib/, pkg name)
    if path_l.startswith("src/") or "/src/" in path_l:
        score += 8
    return score


def detect_entrypoints(files: list[FileFacts], root: Path) -> list[str]:
    hits = [
        f.path
        for f in files
        if Path(f.path).name in ENTRYPOINT_NAMES
        and "test" not in f.path.lower()
    ]
    # Also look for package __init__ that re-exports, and top-level scripts
    for f in files:
        if Path(f.path).name == "__init__.py" and f.path.count("/") <= 2:
            if "test" in f.path.lower():
                continue
            if f.path not in hits:
                hits.append(f.path)
    # Manifests at repo root (case-insensitive dedupe for macOS)
    seen_lower: set[str] = {h.lower() for h in hits}
    for name in MANIFEST_NAMES:
        p = root / name
        if p.exists() and name.lower() not in seen_lower:
            hits.insert(0, name)
            seen_lower.add(name.lower())
    return hits[:20]


def rank_key_files(
    files: list[FileFacts],
    edges: list[ImportEdge],
    limit: int = 20,
) -> list[str]:
    """Rank files by structural importance using a simple import graph."""
    G = nx.DiGraph()
    path_set = {f.path for f in files}
    for f in files:
        G.add_node(f.path)
    for e in edges:
        # Only keep edges that resolve to known files (loose match by basename)
        if e.target in path_set:
            G.add_edge(e.source, e.target)
        else:
            # Relative / package-ish targets: match basename
            target_base = e.target.replace(".", "/").split("/")[-1]
            for p in path_set:
                if Path(p).stem == target_base or Path(p).name == target_base:
                    G.add_edge(e.source, p)
                    break

    def is_noise(path: str) -> bool:
        p = path.lower()
        return any(
            tok in p
            for tok in ("/test", "test_", "/spec", "/example", "/fixture", "docs/")
        )

    primary: list[tuple[float, str]] = []
    secondary: list[tuple[float, str]] = []
    for f in files:
        item = (_score_file(f, G.in_degree(f.path), G.out_degree(f.path)), f.path)
        (secondary if is_noise(f.path) else primary).append(item)
    primary.sort(reverse=True)
    secondary.sort(reverse=True)
    # Prefer non-test / non-example sources for the guided tour
    ranked = [p for _, p in primary] + [p for _, p in secondary]
    return ranked[:limit]


def build_call_graph_summary(edges: list[ImportEdge], top_n: int = 30) -> list[dict]:
    """Collapse edges into a short adjacency list for prompts."""
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        # Keep short targets to control prompt size
        tgt = e.target if len(e.target) < 80 else e.target[:77] + "…"
        adj[e.source].add(tgt)
    rows = sorted(adj.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_n]
    return [{"file": src, "imports": sorted(imps)[:15]} for src, imps in rows]


def pack_llm_context(repo_map: RepoMap, max_chars: int = 24_000) -> str:
    """Compact textual context for the prompt strategy engine."""
    by_path = {f.path: f for f in repo_map.files}
    key_briefs = []
    for path in repo_map.key_files[:18]:
        f = by_path.get(path)
        if not f:
            key_briefs.append({"path": path, "note": "manifest / non-parsed"})
            continue
        key_briefs.append(
            {
                "path": f.path,
                "language": f.language,
                "lines": f.lines,
                "classes": f.classes[:8],
                "functions": f.functions[:12],
                "imports": f.imports[:12],
                "docstring": (f.docstring or "")[:240] or None,
            }
        )

    payload = {
        "repo_url": repo_map.repo_url,
        "repo_name": repo_map.repo_name,
        "languages": repo_map.language_counts,
        "entrypoints": repo_map.entrypoints,
        "folder_tree": repo_map.tree_text[:4000],
        "key_files": key_briefs,
        "import_graph_top": build_call_graph_summary(repo_map.edges),
        "file_count": len(repo_map.files),
    }
    text = json.dumps(payload, indent=2)
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n… [truncated]"
    return text


def build_repo_map(
    repo_url: str,
    root: Path,
    settings: Settings | None = None,
) -> RepoMap:
    """Run Stage 1b + Stage 2 and return a filled RepoMap."""
    settings = settings or Settings()
    _, owner, name = normalize_repo_url(repo_url)
    display_url = f"https://github.com/{owner}/{name}"

    console.print("[cyan]🗺[/] Building structural map…")
    files, edges, lang_counts = parse_repository(root, settings)
    tree = build_tree_text(root)
    entrypoints = detect_entrypoints(files, root)
    key_files = rank_key_files(files, edges)

    # Ensure entrypoints appear in key_files
    for ep in entrypoints:
        if ep not in key_files:
            key_files.insert(0, ep)
    key_files = list(dict.fromkeys(key_files))[:25]

    repo_map = RepoMap(
        repo_url=display_url,
        repo_name=f"{owner}/{name}",
        root=root,
        tree_text=tree,
        files=files,
        edges=edges,
        language_counts=lang_counts,
        entrypoints=entrypoints,
        key_files=key_files,
        metadata={
            "owner": owner,
            "name": name,
            "file_count": len(files),
            "edge_count": len(edges),
        },
    )
    console.print(
        f"[green]✓[/] Map ready — "
        f"{len(files)} files, top keys: {', '.join(key_files[:5])}"
    )
    return repo_map
