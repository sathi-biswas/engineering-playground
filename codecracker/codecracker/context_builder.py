"""Stage 2 — Context Builder & Structural Map.

Builds a folder tree, ranks key files / entrypoints, and packs a compact
context blob for the LLM prompt strategy engine.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Token Budgeting Engine (tiktoken)
# ---------------------------------------------------------------------------


@dataclass
class TokenBudgetReport:
    """Result of fitting a structural payload under a token ceiling."""

    text: str
    token_count: int
    token_budget: int
    encoding_name: str
    trimmed: bool
    trim_steps: list[str] = field(default_factory=list)
    sections_kept: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "token_count": self.token_count,
            "token_budget": self.token_budget,
            "encoding_name": self.encoding_name,
            "trimmed": self.trimmed,
            "trim_steps": list(self.trim_steps),
            "sections_kept": dict(self.sections_kept),
        }


class TokenBudgetingEngine:
    """
    Tiktoken-backed token counter that enforces a max-token threshold
    *before* the structural map is serialized into the LLM user prompt.

    Shrinks payload sections in priority order (graph → tree → key briefs)
    until ``count_tokens(serialized) <= budget``.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 6000,
        encoding_name: str = "cl100k_base",
        output_reserve: int = 0,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        # Usable budget for the structural JSON (caller may reserve for reply)
        self.max_tokens = max(256, max_tokens - max(0, output_reserve))
        self.encoding_name = encoding_name
        self._encoder = self._load_encoder(encoding_name)

    @staticmethod
    def _load_encoder(encoding_name: str):
        try:
            import tiktoken

            try:
                return tiktoken.get_encoding(encoding_name)
            except Exception:  # noqa: BLE001
                return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            console.print(
                "[yellow]Warning:[/] tiktoken not installed — "
                "falling back to ~4 chars/token estimates"
            )
            return None

    def count_tokens(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        # Fallback mirrors OpenAI's rough rule of thumb
        return max(1, (len(text) + 3) // 4)

    def fits(self, text: str) -> bool:
        return self.count_tokens(text) <= self.max_tokens

    def serialize(self, payload: dict) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def fit_payload(self, payload: dict) -> TokenBudgetReport:
        """
        Return serialized JSON guaranteed (best-effort) under ``max_tokens``.

        Trim order (least → most destructive):
          1. Shrink import_graph_top rows / imports-per-row
          2. Truncate folder_tree lines
          3. Drop lowest-priority key_files and shrink symbol lists
          4. Drop optional docstring / imports fields
          5. Hard-truncate serialized text as last resort
        """
        working = json.loads(json.dumps(payload))  # deep copy via JSON
        steps: list[str] = []
        text = self.serialize(working)
        tokens = self.count_tokens(text)

        if tokens <= self.max_tokens:
            return TokenBudgetReport(
                text=text,
                token_count=tokens,
                token_budget=self.max_tokens,
                encoding_name=self.encoding_name,
                trimmed=False,
                sections_kept=self._section_sizes(working),
            )

        # --- progressive shrinks ---
        graph = working.get("import_graph_top") or []
        for limit, per_row in ((20, 10), (12, 6), (6, 4), (3, 3), (0, 0)):
            if tokens <= self.max_tokens:
                break
            if limit == 0:
                working["import_graph_top"] = []
                steps.append("drop import_graph_top")
            else:
                working["import_graph_top"] = [
                    {"file": row.get("file"), "imports": (row.get("imports") or [])[:per_row]}
                    for row in graph[:limit]
                ]
                steps.append(f"import_graph_top→{limit}x{per_row}")
            text = self.serialize(working)
            tokens = self.count_tokens(text)

        tree = working.get("folder_tree") or ""
        for max_lines in (80, 40, 20, 10, 0):
            if tokens <= self.max_tokens:
                break
            if max_lines == 0:
                working["folder_tree"] = ""
                steps.append("drop folder_tree")
            else:
                lines = tree.splitlines()[:max_lines]
                working["folder_tree"] = "\n".join(lines)
                if len(tree.splitlines()) > max_lines:
                    working["folder_tree"] += "\n…"
                steps.append(f"folder_tree→{max_lines} lines")
            text = self.serialize(working)
            tokens = self.count_tokens(text)

        keys = list(working.get("key_files") or [])
        for keep in (12, 8, 5, 3, 1):
            if tokens <= self.max_tokens:
                break
            trimmed_keys = []
            for item in keys[:keep]:
                brief = dict(item)
                brief["classes"] = (brief.get("classes") or [])[:4]
                brief["functions"] = (brief.get("functions") or [])[:6]
                brief["imports"] = (brief.get("imports") or [])[:4]
                doc = brief.get("docstring")
                if isinstance(doc, str):
                    brief["docstring"] = doc[:120]
                trimmed_keys.append(brief)
            working["key_files"] = trimmed_keys
            steps.append(f"key_files→{keep} (shrunk symbols)")
            text = self.serialize(working)
            tokens = self.count_tokens(text)

        # Strip verbose fields if still over
        if tokens > self.max_tokens:
            for item in working.get("key_files") or []:
                item.pop("docstring", None)
                item.pop("imports", None)
                item["classes"] = (item.get("classes") or [])[:2]
                item["functions"] = (item.get("functions") or [])[:3]
            steps.append("strip docstrings/imports from key_files")
            text = self.serialize(working)
            tokens = self.count_tokens(text)

        # Hard truncate as last resort (keeps valid-ish prefix for the model)
        if tokens > self.max_tokens:
            text = self._hard_truncate(text, self.max_tokens)
            tokens = self.count_tokens(text)
            steps.append("hard_truncate serialized JSON")
            working["_truncated"] = True

        console.print(
            f"[cyan]✂[/] Token budget: {tokens}/{self.max_tokens} "
            f"({self.encoding_name}); steps={steps or ['none']}"
        )
        return TokenBudgetReport(
            text=text,
            token_count=tokens,
            token_budget=self.max_tokens,
            encoding_name=self.encoding_name,
            trimmed=True,
            trim_steps=steps,
            sections_kept=self._section_sizes(working),
        )

    def _hard_truncate(self, text: str, budget: int) -> str:
        """Binary-search a prefix that fits under ``budget`` tokens."""
        if self.count_tokens(text) <= budget:
            return text
        marker = "\n… [truncated to token budget]"
        lo, hi = 0, len(text)
        best = marker
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid] + marker
            if self.count_tokens(candidate) <= budget:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    @staticmethod
    def _section_sizes(payload: dict) -> dict[str, int]:
        return {
            "key_files": len(payload.get("key_files") or []),
            "import_graph_top": len(payload.get("import_graph_top") or []),
            "folder_tree_chars": len(payload.get("folder_tree") or ""),
        }


def build_llm_payload(
    repo_map: RepoMap,
    *,
    max_key_files: int = 18,
    max_tree_chars: int = 4000,
    graph_top_n: int = 30,
) -> dict:
    """Assemble the full (pre-budget) structural context dict."""
    by_path = {f.path: f for f in repo_map.files}
    key_briefs = []
    for path in repo_map.key_files[:max_key_files]:
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

    return {
        "repo_url": repo_map.repo_url,
        "repo_name": repo_map.repo_name,
        "languages": repo_map.language_counts,
        "entrypoints": repo_map.entrypoints,
        "folder_tree": repo_map.tree_text[:max_tree_chars],
        "key_files": key_briefs,
        "import_graph_top": build_call_graph_summary(repo_map.edges, top_n=graph_top_n),
        "file_count": len(repo_map.files),
    }


def pack_llm_context(
    repo_map: RepoMap,
    max_chars: int | None = None,
    *,
    settings: Settings | None = None,
    budget_engine: TokenBudgetingEngine | None = None,
) -> str:
    """
    Compact textual context for the prompt strategy engine.

    Prefer tiktoken token budgeting (``TokenBudgetingEngine``). ``max_chars``
    remains as a legacy hard cap applied after token fitting.
    """
    settings = settings or Settings()
    engine = budget_engine or TokenBudgetingEngine(
        max_tokens=settings.llm_context_token_budget,
        encoding_name=settings.tiktoken_encoding,
        # Leave headroom when a combined window is implied via reserve.
        output_reserve=settings.llm_output_token_reserve
        if settings.llm_output_token_reserve > 0
        and settings.llm_output_token_reserve
        < settings.llm_context_token_budget
        else 0,
    )

    payload = build_llm_payload(repo_map)
    report = engine.fit_payload(payload)

    # Stash budget metadata for report.json / debugging
    repo_map.metadata["token_budget"] = report.as_dict()

    text = report.text
    # Optional legacy char ceiling (disabled when None)
    if max_chars is not None and len(text) > max_chars:
        text = text[: max_chars - 20] + "\n… [truncated]"
        repo_map.metadata.setdefault("token_budget", {})["char_truncated"] = True

    if report.trimmed:
        console.print(
            f"[green]✓[/] Context fitted to "
            f"{report.token_count}/{report.token_budget} tokens "
            f"via {report.encoding_name}"
        )
    else:
        console.print(
            f"[dim]Token budget OK: {report.token_count}/{report.token_budget} "
            f"({report.encoding_name})[/]"
        )
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
