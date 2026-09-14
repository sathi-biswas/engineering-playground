"""Stage 1a — clone (or refresh) a public GitHub repository."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from git import Repo
from rich.console import Console

from .config import Settings

console = Console()

_GITHUB_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/#?]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)


def normalize_repo_url(url: str) -> tuple[str, str, str]:
    """Return (canonical_https_url, owner, repo_name)."""
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    match = _GITHUB_RE.match(url)
    if match:
        owner, name = match.group(1), match.group(2)
        return f"https://github.com/{owner}/{name}.git", owner, name

    parsed = urlparse(url if "://" in url else f"https://{url}")
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2:
        owner, name = parts[0], parts[1].removesuffix(".git")
        host = parsed.netloc or "github.com"
        return f"https://{host}/{owner}/{name}.git", owner, name

    raise ValueError(
        f"Could not parse GitHub URL: {url!r}. "
        "Expected e.g. https://github.com/owner/repo"
    )


def clone_repo(url: str, settings: Settings | None = None) -> Path:
    """
    Shallow-clone a public repo into settings.work_dir/<owner>__<repo>.

    Reuses an existing clone when present (fetch + reset to origin/HEAD).
    """
    settings = settings or Settings()
    settings.ensure_dirs()

    canonical, owner, name = normalize_repo_url(url)
    dest = settings.work_dir / f"{owner}__{name}"

    if dest.exists() and (dest / ".git").exists():
        console.print(f"[cyan]↻[/] Reusing clone at [bold]{dest}[/]")
        repo = Repo(dest)
        try:
            repo.remotes.origin.fetch(depth=1 if settings.shallow_clone else None)
            # Stay on whatever default branch origin/HEAD points to
            head_ref = repo.git.rev_parse("--abbrev-ref", "origin/HEAD")
            branch = head_ref.split("/")[-1]
            repo.git.checkout(branch)
            repo.git.reset("--hard", f"origin/{branch}")
        except Exception as exc:  # noqa: BLE001 — best-effort refresh
            console.print(f"[yellow]Warning:[/] could not refresh clone ({exc})")
        return dest

    console.print(f"[cyan]↓[/] Cloning [bold]{canonical}[/] → {dest}")
    kwargs: dict = {"to_path": str(dest)}
    if settings.shallow_clone:
        kwargs["multi_options"] = ["--depth=1"]
    Repo.clone_from(canonical, **kwargs)
    console.print(f"[green]✓[/] Clone ready: {dest}")
    return dest
