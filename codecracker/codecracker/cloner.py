"""Stage 1a — safely clone (or refresh) a public GitHub repository.

Hardening goals
---------------
* Disable git hooks during clone/fetch/checkout (``core.hooksPath`` → os.devnull)
  so a malicious repo cannot RCE via ``post-checkout`` / ``post-merge`` hooks.
* Sanitize owner/repo path segments and contain the clone directory under
  ``settings.work_dir`` to block path-traversal via crafted names.
* Prefer reusing existing clones and enforce a local cooldown between fresh
  clones to reduce unauthenticated GitHub rate-limit pressure.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from git import Repo
from rich.console import Console

from .config import Settings

console = Console()

# GitHub owner / repo slug: letters, digits, dot, underscore, hyphen only.
_SAFE_SLUG = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_GITHUB_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/#?]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)
_ALLOWED_HOSTS = frozenset({"github.com", "www.github.com"})


def _safe_git_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Env that forces hooks off for every git subprocess (Git ≥ 2.31)."""
    env = dict(base or os.environ)
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "core.hooksPath"
    env["GIT_CONFIG_VALUE_0"] = os.devnull
    return env


def _hooks_null_config() -> str:
    """``git -c core.hooksPath=<devnull>`` value (cross-platform)."""
    return f"core.hooksPath={os.devnull}"


def _git_config_env() -> dict[str, str]:
    """Subset of env vars to overlay via ``Repo.git.custom_environment``."""
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": os.devnull,
    }


def sanitize_slug(value: str, *, kind: str = "name") -> str:
    """Reject path traversal / separator injection in owner or repo names."""
    value = (value or "").strip().removesuffix(".git")
    if not value or value in {".", ".."}:
        raise ValueError(f"Invalid repository {kind}: {value!r}")
    if any(sep in value for sep in ("/", "\\", "\x00")):
        raise ValueError(
            f"Invalid repository {kind} (path separators not allowed): {value!r}"
        )
    if ".." in value:
        raise ValueError(f"Invalid repository {kind} (traversal): {value!r}")
    if not _SAFE_SLUG.match(value):
        raise ValueError(
            f"Invalid repository {kind}: {value!r}. "
            "Only alphanumeric characters, '.', '_' and '-' are allowed."
        )
    return value


def normalize_repo_url(url: str) -> tuple[str, str, str]:
    """Return (canonical_https_url, owner, repo_name) for github.com only."""
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    match = _GITHUB_RE.match(url)
    if match:
        owner = sanitize_slug(match.group(1), kind="owner")
        name = sanitize_slug(match.group(2), kind="repo")
        return f"https://github.com/{owner}/{name}.git", owner, name

    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"Only public github.com URLs are supported (got host {host!r}). "
            "Expected e.g. https://github.com/owner/repo"
        )
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"Could not parse GitHub URL: {url!r}. "
            "Expected e.g. https://github.com/owner/repo"
        )
    owner = sanitize_slug(parts[0], kind="owner")
    name = sanitize_slug(parts[1].removesuffix(".git"), kind="repo")
    return f"https://github.com/{owner}/{name}.git", owner, name


def safe_clone_dest(work_dir: Path, owner: str, name: str) -> Path:
    """Build ``work_dir/<owner>__<name>`` and ensure it cannot escape work_dir."""
    owner = sanitize_slug(owner, kind="owner")
    name = sanitize_slug(name, kind="repo")
    root = work_dir.resolve()
    dest = (root / f"{owner}__{name}").resolve()
    try:
        dest.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing clone path outside work dir: {dest} (root={root})"
        ) from exc
    return dest


def _authenticated_clone_url(canonical: str, token: str | None) -> str:
    """Embed a GitHub token for higher git rate limits when provided."""
    if not token:
        return canonical
    parsed = urlparse(canonical)
    netloc = f"x-access-token:{token}@{parsed.hostname}"
    return urlunparse(parsed._replace(netloc=netloc))


def _rate_limit_path(work_dir: Path) -> Path:
    return work_dir / ".clone_rate_limit"


def enforce_clone_rate_limit(settings: Settings, *, is_fresh_clone: bool) -> None:
    """
    Throttle *fresh* clones to avoid burning unauthenticated GitHub IP quotas.

    Reuses of existing checkouts skip the cooldown (local disk only).
    """
    if not is_fresh_clone:
        return
    interval = max(0, int(settings.min_clone_interval_seconds))
    if interval == 0:
        return

    stamp_file = _rate_limit_path(settings.work_dir)
    now = time.time()
    if stamp_file.exists():
        try:
            last = float(stamp_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            last = 0.0
        elapsed = now - last
        if elapsed < interval:
            wait = interval - elapsed
            console.print(
                f"[yellow]⏳[/] Clone rate-limit: waiting {wait:.1f}s "
                f"(min interval {interval}s between fresh clones)"
            )
            time.sleep(wait)

    settings.work_dir.mkdir(parents=True, exist_ok=True)
    stamp_file.write_text(str(time.time()), encoding="utf-8")


def _disable_hooks_in_repo(repo: Repo) -> None:
    """Persist hooksPath=devnull in the local repo config after clone/refresh."""
    with repo.config_writer(config_level="repository") as writer:
        writer.set_value("core", "hooksPath", os.devnull)


def _clone_multi_options(settings: Settings) -> list[str]:
    """git-clone flags including ``-c core.hooksPath=…``."""
    opts = ["-c", _hooks_null_config()]
    if settings.shallow_clone:
        opts.append("--depth=1")
    # Partial clone: skip blob bodies until needed (lighter on rate limits / disk)
    if settings.clone_filter_blob_none:
        opts.append("--filter=blob:none")
    return opts


def _refresh_existing_clone(repo: Repo, settings: Settings) -> None:
    """Fetch + hard-reset with hooks disabled for every git invocation."""
    _disable_hooks_in_repo(repo)
    c_flag = _hooks_null_config()
    with repo.git.custom_environment(**_git_config_env()):
        fetch_cmd = ["git", "-c", c_flag, "fetch", "origin"]
        if settings.shallow_clone:
            fetch_cmd.extend(["--depth", "1"])
        repo.git.execute(fetch_cmd)

        head_ref = repo.git.execute(
            ["git", "-c", c_flag, "rev-parse", "--abbrev-ref", "origin/HEAD"]
        )
        branch = str(head_ref).strip().split("/")[-1]
        repo.git.execute(["git", "-c", c_flag, "checkout", branch])
        repo.git.execute(
            ["git", "-c", c_flag, "reset", "--hard", f"origin/{branch}"]
        )


def clone_repo(url: str, settings: Settings | None = None) -> Path:
    """
    Shallow-clone a public GitHub repo into settings.work_dir/<owner>__<repo>.

    Reuses an existing clone when present (fetch + reset to origin/HEAD).
    Hooks are disabled for all git operations.
    """
    settings = settings or Settings()
    settings.ensure_dirs()

    canonical, owner, name = normalize_repo_url(url)
    dest = safe_clone_dest(settings.work_dir, owner, name)

    if dest.exists() and (dest / ".git").exists():
        console.print(f"[cyan]↻[/] Reusing clone at [bold]{dest}[/]")
        repo = Repo(dest)
        try:
            _refresh_existing_clone(repo, settings)
        except Exception as exc:  # noqa: BLE001 — best-effort refresh
            console.print(f"[yellow]Warning:[/] could not refresh clone ({exc})")
        return dest

    enforce_clone_rate_limit(settings, is_fresh_clone=True)

    clone_url = _authenticated_clone_url(canonical, settings.github_token)
    console.print(f"[cyan]↓[/] Cloning [bold]{canonical}[/] → {dest}")
    if settings.github_token:
        console.print(
            "[dim]Using GITHUB_TOKEN for authenticated clone (higher rate limits)[/]"
        )

    # allow_unsafe_options: GitPython treats ``-c`` as unsafe; required for hooksPath.
    Repo.clone_from(
        clone_url,
        to_path=str(dest),
        env=_safe_git_env(),
        multi_options=_clone_multi_options(settings),
        allow_unsafe_options=True,
    )
    _disable_hooks_in_repo(Repo(dest))
    console.print(f"[green]✓[/] Clone ready (hooks disabled): {dest}")
    return dest
