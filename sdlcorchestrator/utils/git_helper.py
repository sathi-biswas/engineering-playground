"""Git branch / commit / push helpers and GitHub Pull Request creation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from github import Github, GithubException

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class GitHelperError(RuntimeError):
    """Raised when a git or GitHub operation fails."""


def _run_git(
    args: list[str],
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a git subcommand in *cwd* and return the completed process."""
    cmd = ["git", *args]
    logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise GitHelperError(
            f"git {' '.join(args)} failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


class GitHelper:
    """Local git operations plus GitHub PR creation via PyGithub."""

    def __init__(
        self,
        repo_path: str | Path,
        github_repo_name: str | None = None,
        token: str | None = None,
    ) -> None:
        settings = get_settings()
        self.repo_path = Path(repo_path).resolve()
        self.github_repo_name = github_repo_name or settings.default_github_repo_name
        secret = token or (
            settings.github_token.get_secret_value() if settings.github_token else None
        )
        self._token = secret
        self._gh: Github | None = Github(secret) if secret else None
        self.base_branch = settings.github_default_base_branch

        if not self.repo_path.exists():
            raise GitHelperError(f"Repository path does not exist: {self.repo_path}")

    # ------------------------------------------------------------------
    # Local git
    # ------------------------------------------------------------------

    def current_branch(self) -> str:
        """Return the currently checked-out branch name."""
        result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], self.repo_path)
        return result.stdout.strip()

    def create_branch(self, branch_name: str, from_branch: str | None = None) -> str:
        """Create and check out *branch_name* from *from_branch* (default: base)."""
        base = from_branch or self.base_branch
        # Ensure we start from a clean base
        _run_git(["fetch", "origin", base], self.repo_path, check=False)
        _run_git(["checkout", base], self.repo_path, check=False)
        _run_git(["pull", "origin", base], self.repo_path, check=False)

        # Delete local branch if it already exists (idempotent for retries)
        existing = _run_git(["branch", "--list", branch_name], self.repo_path)
        if existing.stdout.strip():
            _run_git(["checkout", base], self.repo_path, check=False)
            _run_git(["branch", "-D", branch_name], self.repo_path, check=False)

        _run_git(["checkout", "-b", branch_name], self.repo_path)
        logger.info("Created branch %s from %s", branch_name, base)
        return branch_name

    def stage_all(self, paths: list[str] | None = None) -> None:
        """Stage files for commit. Stages all changes when *paths* is None."""
        if paths:
            for p in paths:
                _run_git(["add", "--", p], self.repo_path)
        else:
            _run_git(["add", "-A"], self.repo_path)

    def commit(self, message: str) -> str:
        """Create a commit with *message*. Returns the new commit SHA."""
        # Configure local identity if missing (CI / sandbox friendliness)
        name = _run_git(["config", "user.name"], self.repo_path, check=False)
        if not name.stdout.strip():
            _run_git(["config", "user.name", "SDLC Orchestrator"], self.repo_path)
            _run_git(["config", "user.email", "sdlc-orchestrator@local"], self.repo_path)

        status = _run_git(["status", "--porcelain"], self.repo_path)
        if not status.stdout.strip():
            logger.warning("Nothing to commit — working tree clean")
            head = _run_git(["rev-parse", "HEAD"], self.repo_path)
            return head.stdout.strip()

        _run_git(["commit", "-m", message], self.repo_path)
        head = _run_git(["rev-parse", "HEAD"], self.repo_path)
        sha = head.stdout.strip()
        logger.info("Committed %s: %s", sha[:8], message.splitlines()[0])
        return sha

    def push(self, branch_name: str, set_upstream: bool = True) -> None:
        """Push *branch_name* to origin."""
        args = ["push"]
        if set_upstream:
            args += ["-u", "origin", branch_name]
        else:
            args += ["origin", branch_name]
        _run_git(args, self.repo_path)
        logger.info("Pushed branch %s to origin", branch_name)

    def get_diff(self, base: str | None = None) -> str:
        """Return unified diff of current branch against *base*."""
        base = base or self.base_branch
        result = _run_git(["diff", f"{base}...HEAD"], self.repo_path, check=False)
        return result.stdout

    # ------------------------------------------------------------------
    # GitHub PR
    # ------------------------------------------------------------------

    def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str | None = None,
        draft: bool = False,
    ) -> dict[str, Any]:
        """Open a GitHub Pull Request and return ``{url, number, html_url}``."""
        if not self._gh:
            raise GitHelperError(
                "GITHUB_TOKEN is not configured — cannot create Pull Request"
            )
        if not self.github_repo_name:
            raise GitHelperError("github_repo_name is required to create a PR")

        base = base_branch or self.base_branch
        try:
            repo = self._gh.get_repo(self.github_repo_name)
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base,
                draft=draft,
            )
            logger.info("Created PR #%s: %s", pr.number, pr.html_url)
            return {
                "url": pr.html_url,
                "html_url": pr.html_url,
                "number": pr.number,
                "id": pr.id,
            }
        except GithubException as exc:
            raise GitHelperError(f"GitHub PR creation failed: {exc.data or exc}") from exc

    def post_pr_review(
        self,
        pr_number: int,
        body: str,
        event: str = "COMMENT",
        comments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Post a PR review.

        Args:
            pr_number: Pull request number.
            body: Overall review summary.
            event: One of ``APPROVE``, ``REQUEST_CHANGES``, ``COMMENT``.
            comments: Optional list of ``{path, line, body}`` inline comments.
        """
        if not self._gh or not self.github_repo_name:
            raise GitHelperError("GitHub credentials / repo name required for review")

        repo = self._gh.get_repo(self.github_repo_name)
        pr = repo.get_pull(pr_number)

        review_comments = []
        for c in comments or []:
            review_comments.append(
                {
                    "path": c["path"],
                    "line": c.get("line", 1),
                    "body": c["body"],
                }
            )

        try:
            if review_comments:
                # Commit SHA required for multi-line review API
                commit = repo.get_commit(pr.head.sha)
                pr.create_review(
                    commit=commit,
                    body=body,
                    event=event,
                    comments=review_comments,
                )
            else:
                pr.create_review(body=body, event=event)
            logger.info("Posted %s review on PR #%s", event, pr_number)
        except GithubException as exc:
            # Fallback: post as a plain issue comment if inline review fails
            logger.warning("Inline review failed (%s); posting comment instead", exc)
            pr.create_issue_comment(f"**Review ({event})**\n\n{body}")

    def get_pr_diff(self, pr_number: int) -> str:
        """Fetch the unified diff text for a pull request."""
        if not self._gh or not self.github_repo_name:
            raise GitHelperError("GitHub credentials / repo name required")
        repo = self._gh.get_repo(self.github_repo_name)
        pr = repo.get_pull(pr_number)
        # PyGithub exposes .diff_url; fetch via files for reliability
        files = pr.get_files()
        chunks: list[str] = []
        for f in files:
            chunks.append(f"--- a/{f.filename}\n+++ b/{f.filename}\n{f.patch or ''}")
        return "\n\n".join(chunks)
