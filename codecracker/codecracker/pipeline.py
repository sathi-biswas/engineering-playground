"""End-to-end pipeline wiring for CodeCracker."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .benchmark import build_benchmark, format_cost, format_seconds
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
    4. Output Generator → README.md (includes runtime benchmark table)
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
    t0 = time.perf_counter()
    if local_path is not None:
        root = Path(local_path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Local path not found: {root}")
        console.print(f"[cyan]📂[/] Using local path [bold]{root}[/]")
    else:
        root = clone_repo(repo_url, settings)
    clone_s = time.perf_counter() - t0

    # Stage 1b + 2 — parse + structural map
    t1 = time.perf_counter()
    repo_map = build_repo_map(repo_url, root, settings)
    map_s = time.perf_counter() - t1

    # Stage 3 — prompt strategy / synthesis
    t2 = time.perf_counter()
    report = synthesize(repo_map, settings, provider=provider)
    synth_s = time.perf_counter() - t2

    # Stage 4 — write README with this-run benchmark table
    t3 = time.perf_counter()
    # Provisional write_s; finalize after measuring the write itself
    benchmark = build_benchmark(
        repo_map,
        report,
        clone_s=clone_s,
        map_s=map_s,
        synth_s=synth_s,
        write_s=0.0,
    )
    dest = write_outputs(
        repo_map,
        report,
        settings,
        out_dir=out_dir,
        benchmark=benchmark,
    )
    write_s = time.perf_counter() - t3

    benchmark = build_benchmark(
        repo_map,
        report,
        clone_s=clone_s,
        map_s=map_s,
        synth_s=synth_s,
        write_s=write_s,
    )
    # Refresh README/JSON once with accurate write timing (cheap local rewrite)
    write_outputs(
        repo_map,
        report,
        settings,
        out_dir=dest,
        benchmark=benchmark,
    )

    console.print(
        Panel.fit(
            f"[green]Done.[/] Open [bold]{dest / 'README.md'}[/] and follow the guided tour.\n"
            f"Benchmark: {benchmark.files_analyzed} files | "
            f"heuristic {format_seconds(benchmark.time_heuristic_s)} | "
            f"LLM {format_seconds(benchmark.time_llm_s)} | "
            f"{format_cost(benchmark.approx_token_cost_usd)}",
            border_style="green",
        )
    )
    return repo_map, report, dest
