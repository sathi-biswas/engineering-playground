"""CLI entrypoint: `python -m codecracker crack <github-url>`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import __version__
from .config import Settings
from .pipeline import run

app = typer.Typer(
    name="codecracker",
    help="Crack open a public GitHub repo: architecture summary + guided file-by-file tour.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"codecracker {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """CodeCracker — hands-on architectural understanding of third-party code."""


@app.command("crack")
def crack(
    repo_url: str = typer.Argument(
        ...,
        help="Public GitHub URL, e.g. https://github.com/pallets/flask",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Synthesis backend: auto | openai | anthropic | heuristic",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Output directory for README.md (default: output/<owner>__<repo>)",
    ),
    local: Optional[Path] = typer.Option(
        None,
        "--local",
        help="Skip clone; analyze an existing local checkout instead",
    ),
    max_files: int = typer.Option(
        400,
        "--max-files",
        help="Cap on source files to parse (keeps big monorepos tractable)",
    ),
    max_tour: int = typer.Option(
        25,
        "--max-tour",
        help="Max steps in the guided file-by-file walkthrough",
    ),
    full_clone: bool = typer.Option(
        False,
        "--full-clone",
        help="Disable shallow clone (slower; use when you need full history)",
    ),
) -> None:
    """Clone, map, synthesize, and write a guided architecture README."""
    settings = Settings(
        max_files=max_files,
        max_tour_steps=max_tour,
        shallow_clone=not full_clone,
    )
    try:
        run(
            repo_url,
            settings=settings,
            provider=provider,
            out_dir=out,
            local_path=local,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("tour")
def tour(
    report_json: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to a previously generated report.json",
    ),
) -> None:
    """Pretty-print a guided tour from an existing report.json (no re-clone)."""
    import json

    data = json.loads(report_json.read_text(encoding="utf-8"))
    console.print(f"[bold]Guided tour — {data.get('repo_name', '?')}[/]\n")
    for step in data.get("guided_tour", []):
        console.print(f"[cyan]Step {step['order']}[/] — {step.get('title')}")
        console.print(f"  file: [bold]{step.get('path')}[/]")
        console.print(f"  why:  {step.get('why')}")
        for item in step.get("what_to_look_for") or []:
            console.print(f"    • {item}")
        if step.get("next_hint"):
            console.print(f"  → {step['next_hint']}")
        console.print()


if __name__ == "__main__":
    app()
