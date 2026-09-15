"""Stage 3 — LLM Prompt Strategy Engine.

Produces Architecture Summary, Mermaid.js flow, and a guided file-by-file tour.
Falls back to a deterministic heuristic synthesizer when no API key is set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console

from .config import Settings
from .context_builder import pack_llm_context
from .models import ArchitectureReport, FileFacts, GuidedStep, RepoMap

console = Console()

SYSTEM_PROMPT = """You are CodeCracker, an expert at onboarding engineers onto unfamiliar \
(legacy / third-party) codebases. Given a structural map (folder tree, key files, \
imports, entrypoints), you produce a clear architecture brief and a *guided walkthrough* \
that tells the reader which file to open next and what to look for.

Respond with STRICT JSON only (no markdown fences) matching this schema:
{
  "overview": "2-4 sentence project purpose",
  "architecture_summary": "markdown subsections describing layers/modules",
  "data_flow": "markdown describing request/data lifecycle",
  "mermaid": "a mermaid flowchart or graph TD describing major components",
  "key_files": [{"path": "...", "role": "one-line why it matters"}],
  "guided_tour": [
    {
      "order": 1,
      "path": "relative/path",
      "title": "short title",
      "why": "why open this file now",
      "what_to_look_for": ["bullet", "bullet"],
      "next_hint": "what to open next and why"
    }
  ],
  "how_to_extend": "markdown tips for contributing / tracing a change"
}

Rules:
- guided_tour should be 8-20 ordered steps, file-by-file, starting at entrypoints.
- Prefer real paths from the provided context; do not invent paths.
- Mermaid must be valid graph/flowchart syntax without ``` fences.
"""


def build_user_prompt(repo_map: RepoMap, settings: Settings | None = None) -> str:
    ctx = pack_llm_context(repo_map, settings=settings)
    return (
        f"Analyze this repository and produce the JSON brief.\n\n"
        f"Repository: {repo_map.repo_url}\n\n"
        f"STRUCTURAL CONTEXT (JSON):\n{ctx}\n"
    )


# ---------------------------------------------------------------------------
# Heuristic fallback (no LLM required)
# ---------------------------------------------------------------------------


def _lang_blurb(counts: dict[str, int]) -> str:
    if not counts:
        return "mixed / unknown"
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{lang} ({n})" for lang, n in ranked[:5])


def _guess_layers(files: list[FileFacts]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {
        "entry / CLI": [],
        "API / HTTP": [],
        "core / domain": [],
        "data / persistence": [],
        "tests": [],
        "config / build": [],
    }

    def _prefer_src(paths: list[str]) -> list[str]:
        return sorted(
            paths,
            key=lambda p: (
                0 if p.startswith("src/") or "/src/" in p else 1,
                0 if "example" not in p.lower() else 1,
                p.count("/"),
                p,
            ),
        )

    for f in files:
        p = f.path.lower()
        name = Path(f.path).name.lower()
        if "test" in p or "spec" in p:
            buckets["tests"].append(f.path)
        elif name in {
            "main.py",
            "__main__.py",
            "cli.py",
            "manage.py",
            "main.go",
            "main.rs",
            "index.js",
            "index.ts",
            "server.js",
            "app.py",
            "app.js",
        }:
            buckets["entry / CLI"].append(f.path)
        elif any(
            x in p
            for x in (
                "api/",
                "routes/",
                "handlers/",
                "controllers/",
                "views/",
                "http/",
                "endpoint",
            )
        ):
            buckets["API / HTTP"].append(f.path)
        elif any(
            x in p
            for x in (
                "model",
                "schema",
                "db/",
                "database",
                "repository",
                "storage",
                "migration",
            )
        ):
            buckets["data / persistence"].append(f.path)
        elif name in {
            "setup.py",
            "pyproject.toml",
            "package.json",
            "cargo.toml",
            "go.mod",
            "dockerfile",
            "makefile",
        }:
            buckets["config / build"].append(f.path)
        elif any(x in p for x in ("docs/", "examples/", "example/", ".github/")):
            continue  # noise for layer summary
        else:
            buckets["core / domain"].append(f.path)

    return {
        k: _prefer_src(v)[:8] for k, v in buckets.items() if v
    }


def _heuristic_mermaid(repo_map: RepoMap, layers: dict[str, list[str]]) -> str:
    lines = ["flowchart TD", f'  ROOT["{repo_map.repo_name}"]']
    node_ids: dict[str, str] = {}
    for i, layer in enumerate(layers):
        nid = f"L{i}"
        node_ids[layer] = nid
        lines.append(f'  {nid}["{layer}"]')
        lines.append(f"  ROOT --> {nid}")
    # Connect entry → api → core → data when present
    order = ["entry / CLI", "API / HTTP", "core / domain", "data / persistence"]
    present = [x for x in order if x in node_ids]
    for a, b in zip(present, present[1:]):
        lines.append(f"  {node_ids[a]} --> {node_ids[b]}")
    return "\n".join(lines)


def _build_guided_tour(repo_map: RepoMap, max_steps: int) -> list[GuidedStep]:
    by_path = {f.path: f for f in repo_map.files}
    ordered: list[str] = []
    for ep in repo_map.entrypoints:
        if ep not in ordered:
            ordered.append(ep)
    for kf in repo_map.key_files:
        if kf not in ordered:
            ordered.append(kf)
    ordered = ordered[:max_steps]

    steps: list[GuidedStep] = []
    for i, path in enumerate(ordered, start=1):
        facts = by_path.get(path)
        name = Path(path).name
        why_bits = []
        look: list[str] = []
        if name in {"README.md", "readme.md"}:
            title = "Orient from the project README"
            why_bits.append("Authors usually document purpose, setup, and module layout here.")
            look = ["Installation / quickstart", "Architecture section if any", "Links to docs/"]
        elif name in {"package.json", "pyproject.toml", "Cargo.toml", "go.mod"}:
            title = "Read the package manifest"
            why_bits.append("Dependencies and entry scripts reveal the tech stack and run surface.")
            look = ["dependencies / scripts", "package name & description", "bin / entry points"]
        elif facts and facts.functions:
            title = f"Inspect `{name}` — primary symbols"
            why_bits.append(
                f"Defines {len(facts.functions)} function(s)"
                + (f" and {len(facts.classes)} class(es)" if facts.classes else "")
                + "; often a hub in the import graph."
            )
            look = [
                f"Functions: {', '.join(facts.functions[:6])}",
                f"Imports: {', '.join(facts.imports[:6]) or '—'}",
            ]
            if facts.classes:
                look.insert(0, f"Classes: {', '.join(facts.classes[:6])}")
            if facts.docstring:
                look.append(f"Module docstring: {facts.docstring[:120]}")
        else:
            title = f"Open `{path}`"
            why_bits.append("Ranked as structurally important by path depth, size, and import centrality.")
            look = ["Skim top-level declarations", "Note who imports this file", "Follow outbound imports"]

        next_path = ordered[i] if i < len(ordered) else None
        next_hint = (
            f"Next: open `{next_path}` and compare how it is reached from `{path}`."
            if next_path
            else "End of tour — try tracing one feature end-to-end from entry to data layer."
        )
        steps.append(
            GuidedStep(
                order=i,
                path=path,
                title=title,
                why=" ".join(why_bits),
                what_to_look_for=look,
                next_hint=next_hint,
            )
        )
    return steps


def synthesize_heuristic(repo_map: RepoMap, settings: Settings) -> ArchitectureReport:
    """Deterministic architecture brief — always available offline."""
    layers = _guess_layers(repo_map.files)
    lang = _lang_blurb(repo_map.language_counts)

    overview = (
        f"**{repo_map.repo_name}** ({repo_map.repo_url}) looks like a "
        f"{lang} codebase with ~{len(repo_map.files)} parsed source files. "
        f"CodeCracker ranked entrypoints at: "
        f"{', '.join(repo_map.entrypoints[:5]) or 'n/a'}."
    )

    arch_parts = ["### Inferred layers\n"]
    for layer, paths in layers.items():
        arch_parts.append(f"- **{layer}**: " + ", ".join(f"`{p}`" for p in paths[:5]))
    arch_parts.append(
        "\n### Folder tree (truncated)\n\n```\n"
        + "\n".join(repo_map.tree_text.splitlines()[:60])
        + "\n```"
    )
    architecture_summary = "\n".join(arch_parts)

    data_flow = (
        "### Suggested mental model\n\n"
        "1. Start at an **entrypoint** (CLI / `main` / HTTP server bootstrap).\n"
        "2. Follow **imports** into API / handlers (if present).\n"
        "3. Drop into **core / domain** modules for business rules.\n"
        "4. Land in **data / persistence** (models, repos, I/O).\n"
        "5. Cross-check with **tests** to see intended behavior.\n\n"
        "Import edges captured: "
        f"**{len(repo_map.edges)}**. Use the guided tour below for a file-by-file path."
    )

    key_files = []
    by_path = {f.path: f for f in repo_map.files}
    for path in repo_map.key_files[:15]:
        f = by_path.get(path)
        if f and f.docstring:
            role = f.docstring.split("\n")[0][:120]
        elif f:
            bits = []
            if f.classes:
                bits.append(f"classes: {', '.join(f.classes[:3])}")
            if f.functions:
                bits.append(f"fns: {', '.join(f.functions[:4])}")
            role = "; ".join(bits) or f"{f.language} source ({f.lines} lines)"
        else:
            role = "project manifest / docs"
        key_files.append({"path": path, "role": role})

    tour = _build_guided_tour(repo_map, settings.max_tour_steps)
    how = (
        "### How to extend understanding\n\n"
        "- Pick one user-facing feature and bisect: entry → handler → domain → storage.\n"
        "- Search for the feature keyword, then open the highest-ranked key file that hits.\n"
        "- Re-run CodeCracker with `--llm` after setting an API key for a richer narrative.\n"
        "- Diff against a known-good commit when onboarding onto legacy forks.\n"
    )

    return ArchitectureReport(
        overview=overview,
        architecture_summary=architecture_summary,
        data_flow=data_flow,
        mermaid=_heuristic_mermaid(repo_map, layers),
        key_files=key_files,
        guided_tour=tour,
        how_to_extend=how,
        raw_prompt=build_user_prompt(repo_map, settings),
        provider="heuristic",
    )


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _report_from_llm_json(
    data: dict[str, Any],
    repo_map: RepoMap,
    settings: Settings,
    provider: str,
    raw_prompt: str,
) -> ArchitectureReport:
    tour_raw = data.get("guided_tour") or []
    steps: list[GuidedStep] = []
    for i, item in enumerate(tour_raw, start=1):
        steps.append(
            GuidedStep(
                order=int(item.get("order", i)),
                path=str(item.get("path", "")),
                title=str(item.get("title", "")),
                why=str(item.get("why", "")),
                what_to_look_for=list(item.get("what_to_look_for") or []),
                next_hint=item.get("next_hint"),
            )
        )
    if not steps:
        steps = _build_guided_tour(repo_map, settings.max_tour_steps)

    key_files = data.get("key_files") or [
        {"path": p, "role": ""} for p in repo_map.key_files[:12]
    ]
    return ArchitectureReport(
        overview=str(data.get("overview", "")),
        architecture_summary=str(data.get("architecture_summary", "")),
        data_flow=str(data.get("data_flow", "")),
        mermaid=str(data.get("mermaid", "")).replace("```mermaid", "").replace("```", "").strip(),
        key_files=[{"path": str(k.get("path", "")), "role": str(k.get("role", ""))} for k in key_files],
        guided_tour=steps,
        how_to_extend=str(data.get("how_to_extend", "")),
        raw_prompt=raw_prompt,
        provider=provider,
    )


def _call_openai(settings: Settings, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or "{}"


def _call_anthropic(settings: Settings, user_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=4096,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return msg.content[0].text  # type: ignore[union-attr]


def resolve_provider(settings: Settings, force: str | None = None) -> str:
    choice = (force or settings.llm_provider or "auto").lower()
    if choice in {"heuristic", "none", "off"}:
        return "heuristic"
    if choice == "openai":
        return "openai" if settings.openai_api_key else "heuristic"
    if choice == "anthropic":
        return "anthropic" if settings.anthropic_api_key else "heuristic"
    # auto
    if settings.openai_api_key:
        return "openai"
    if settings.anthropic_api_key:
        return "anthropic"
    return "heuristic"


def synthesize(
    repo_map: RepoMap,
    settings: Settings | None = None,
    provider: str | None = None,
) -> ArchitectureReport:
    """Stage 3 entrypoint — LLM when available, else heuristic."""
    settings = settings or Settings()
    raw_prompt = build_user_prompt(repo_map, settings)
    chosen = resolve_provider(settings, provider)

    if chosen == "heuristic":
        console.print(
            "[yellow]◆[/] No LLM key (or --provider heuristic) — "
            "using structural heuristic synthesizer"
        )
        return synthesize_heuristic(repo_map, settings)

    console.print(f"[cyan]◆[/] Prompt strategy engine via [bold]{chosen}[/]…")
    try:
        if chosen == "openai":
            text = _call_openai(settings, raw_prompt)
        else:
            text = _call_anthropic(settings, raw_prompt)
        data = _extract_json(text)
        report = _report_from_llm_json(data, repo_map, settings, chosen, raw_prompt)
        console.print(f"[green]✓[/] LLM synthesis ready ({len(report.guided_tour)} tour steps)")
        return report
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Warning:[/] LLM failed ({exc}); falling back to heuristic")
        report = synthesize_heuristic(repo_map, settings)
        report.raw_prompt = raw_prompt
        return report
