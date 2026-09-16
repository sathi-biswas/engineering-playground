"""Isolated subprocess runner for executing unit tests (pytest)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TestRunResult:
    """Outcome of a single test-suite invocation."""

    passed: bool
    output: str
    exit_code: int
    command: list[str]
    duration_hint: str = ""


class TestRunner:
    """Execute pytest (or a custom command) inside a target repository."""

    def __init__(
        self,
        repo_path: str | Path,
        timeout: int | None = None,
        extra_args: str | None = None,
    ) -> None:
        settings = get_settings()
        self.repo_path = Path(repo_path).resolve()
        self.timeout = timeout or settings.test_timeout_seconds
        self.extra_args = (extra_args or settings.pytest_args).split()

        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repo path not found: {self.repo_path}")

    def run_pytest(
        self,
        test_paths: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> TestRunResult:
        """Run pytest in an isolated subprocess.

        Args:
            test_paths: Optional specific test files/dirs. Defaults to full suite.
            env_overrides: Extra environment variables for the child process.
        """
        cmd = ["python", "-m", "pytest", *self.extra_args]
        if test_paths:
            cmd.extend(test_paths)

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if env_overrides:
            env.update(env_overrides)

        logger.info("Running tests: %s (cwd=%s, timeout=%ss)", " ".join(cmd), self.repo_path, self.timeout)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")
            passed = result.returncode == 0
            logger.info("Tests %s (exit=%s)", "PASSED" if passed else "FAILED", result.returncode)
            return TestRunResult(
                passed=passed,
                output=output[-50000:],  # cap log size
                exit_code=result.returncode,
                command=cmd,
            )
        except subprocess.TimeoutExpired as exc:
            partial = ""
            if exc.stdout:
                partial += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode()
            if exc.stderr:
                partial += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode()
            logger.error("Test run timed out after %ss", self.timeout)
            return TestRunResult(
                passed=False,
                output=f"TIMEOUT after {self.timeout}s\n{partial}",
                exit_code=-1,
                command=cmd,
                duration_hint=f"timeout>{self.timeout}s",
            )
        except FileNotFoundError:
            msg = "pytest / python executable not found in PATH"
            logger.error(msg)
            return TestRunResult(
                passed=False,
                output=msg,
                exit_code=127,
                command=cmd,
            )

    def run_command(self, command: list[str]) -> TestRunResult:
        """Run an arbitrary test command (e.g. ``npm test``, ``go test``)."""
        logger.info("Running custom test command: %s", " ".join(command))
        try:
            result = subprocess.run(
                command,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")
            return TestRunResult(
                passed=result.returncode == 0,
                output=output[-50000:],
                exit_code=result.returncode,
                command=command,
            )
        except subprocess.TimeoutExpired:
            return TestRunResult(
                passed=False,
                output=f"TIMEOUT after {self.timeout}s",
                exit_code=-1,
                command=command,
            )
