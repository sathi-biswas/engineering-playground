"""End-to-end pipeline wiring for CodeCracker."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .cloner import clone_repo
from .config import Settings
from .context_builder import build_repo_map
from .llm_engine import synthesize
from .models import ArchitectureReport, RepoMap
from .output_generator import write_outputs

console = Console()


def run(
    repo_url: str,
    *,
    settings: Settings | None = None,
    provider: str | None = None,
    out_dir: Path | None = None,
    local_path: Path | None = None,
) -> tuple[RepoMap, ArchitectureReport, Path]:
    """
    Execute the 4-stage architecture understanding pipeline.

    1. Git Cloner & AST Parser
    2. Context Builder & Structural Map
    3. LLM Prompt Strategy Engine
    4. Output Generator → README.md
    """
    settings = settings or Settings()
    settings.ensure_dirs()

    console.print(
        Panel.fit(
            "[bold]CodeCracker[/] — guided codebase architecture walkthrough",
            border_style="cyan",
        )
    )

    # Stage 1a — clone (or use local checkout)
    if local_path is not None:
        root = Path(local_path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Local path not found: {root}")
        console.print(f"[cyan]📂[/] Using local path [bold]{root}[/]")
    else:
        root = clone_repo(repo_url, settings)

    # Stage 1b + 2 — parse + structural map
    repo_map = build_repo_map(repo_url, root, settings)

    # Stage 3 — prompt strategy / synthesis
    report = synthesize(repo_map, settings, provider=provider)

    # Stage 4 — write README
    dest = write_outputs(repo_map, report, settings, out_dir=out_dir)

    console.print(
        Panel.fit(
            f"[green]Done.[/] Open [bold]{dest / 'README.md'}[/] and follow the guided tour.",
            border_style="green",
        )
    )
    return repo_map, report, dest
