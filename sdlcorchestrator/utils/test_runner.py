"""Isolated subprocess runner for executing unit tests (pytest).

Prefers the *target repository's* Python interpreter (its virtualenv) so
dependencies like ``rich`` / ``jinja2`` / ``networkx`` resolve correctly.
Falls back to the orchestrator interpreter when no target venv is found.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Relative candidates under the target repo root (Unix + Windows layouts)
_VENV_PYTHON_CANDIDATES = (
    ".venv/bin/python",
    ".venv/bin/python3",
    "venv/bin/python",
    "venv/bin/python3",
    "codecracker_venv/bin/python",
    "codecracker_venv/bin/python3",
    ".venv/Scripts/python.exe",
    "venv/Scripts/python.exe",
    "codecracker_venv/Scripts/python.exe",
)


@dataclass
class TestRunResult:
    """Outcome of a single test-suite invocation."""

    passed: bool
    output: str
    exit_code: int
    command: list[str]
    duration_hint: str = ""
    python_bin: str = ""
    used_fallback: bool = False


def resolve_target_python(
    repo_path: Path,
    explicit: str | Path | None = None,
) -> tuple[Path, bool]:
    """Resolve the Python executable used to run pytest.

    Priority:
      1. Explicit path (constructor arg or ``TARGET_PYTHON_BIN``)
      2. Auto-detect common venv layouts under *repo_path*
      3. Fail-safe: ``sys.executable`` (orchestrator / current interpreter)

    Returns:
        ``(python_path, used_fallback)``

    Note: venv ``bin/python`` is often a symlink to the base interpreter.
    We keep the path *inside the venv* (``absolute()``, not symlink-followed
    ``resolve()``) so site-packages from that venv remain active at runtime.
    """
    if explicit:
        candidate = Path(str(explicit)).expanduser()
        if not candidate.is_absolute():
            repo_rel = (repo_path / candidate).absolute()
            if repo_rel.exists():
                candidate = repo_rel
            else:
                candidate = candidate.absolute()
        else:
            candidate = candidate.absolute()
        if candidate.exists():
            logger.info("Using configured target Python: %s", candidate)
            return candidate, False
        logger.warning(
            "TARGET_PYTHON_BIN=%s not found — trying auto-detect / fail-safe",
            explicit,
        )

    for rel in _VENV_PYTHON_CANDIDATES:
        candidate = (repo_path / rel).absolute()
        if candidate.exists():
            logger.info("Auto-detected target venv Python: %s", candidate)
            return candidate, False

    fallback = Path(sys.executable)
    logger.warning(
        "No target venv Python found under %s (looked for %s). "
        "Fail-safe: using orchestrator interpreter %s — install deps into a "
        "repo-local .venv or set TARGET_PYTHON_BIN to avoid ModuleNotFoundError.",
        repo_path,
        ", ".join(_VENV_PYTHON_CANDIDATES[:6]) + ", …",
        fallback,
    )
    return fallback, True


class TestRunner:
    """Execute pytest via the target repo's Python (venv) when available."""

    def __init__(
        self,
        repo_path: str | Path,
        timeout: int | None = None,
        extra_args: str | None = None,
        python_bin: str | Path | None = None,
    ) -> None:
        settings = get_settings()
        self.repo_path = Path(repo_path).resolve()
        self.timeout = timeout or settings.test_timeout_seconds
        self.extra_args = (extra_args or settings.pytest_args).split()
        explicit = python_bin if python_bin is not None else settings.target_python_bin
        self.python_bin, self.used_fallback = resolve_target_python(
            self.repo_path, explicit=explicit
        )

        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repo path not found: {self.repo_path}")

    def run_pytest(
        self,
        test_paths: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> TestRunResult:
        """Run ``{target_python} -m pytest`` in an isolated subprocess.

        Using ``python -m pytest`` (not a bare ``pytest`` on PATH) ensures
        packages installed in the target virtualenv are importable.
        """
        paths: Sequence[str] = test_paths if test_paths else []
        cmd = [
            str(self.python_bin),
            "-m",
            "pytest",
            *self.extra_args,
            *paths,
        ]

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Prefer the target interpreter's site-packages over ambient PYTHONPATH
        env.pop("PYTHONPATH", None)
        if env_overrides:
            env.update(env_overrides)

        logger.info(
            "Running tests: %s (cwd=%s, timeout=%ss, fallback=%s)",
            " ".join(cmd),
            self.repo_path,
            self.timeout,
            self.used_fallback,
        )

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
            output = self._annotate_missing_deps(output, result.returncode)
            passed = result.returncode == 0
            logger.info("Tests %s (exit=%s)", "PASSED" if passed else "FAILED", result.returncode)
            return TestRunResult(
                passed=passed,
                output=output[-50000:],
                exit_code=result.returncode,
                command=cmd,
                python_bin=str(self.python_bin),
                used_fallback=self.used_fallback,
            )
        except subprocess.TimeoutExpired as exc:
            partial = ""
            if getattr(exc, "stdout", None):
                partial += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode()
            if getattr(exc, "stderr", None):
                partial += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode()
            logger.error("Test run timed out after %ss", self.timeout)
            return TestRunResult(
                passed=False,
                output=f"TIMEOUT after {self.timeout}s\n{partial}",
                exit_code=-1,
                command=cmd,
                duration_hint=f"timeout>{self.timeout}s",
                python_bin=str(self.python_bin),
                used_fallback=self.used_fallback,
            )
        except FileNotFoundError:
            msg = (
                f"Python executable not found: {self.python_bin}. "
                "Create a venv under the target repo or set TARGET_PYTHON_BIN."
            )
            logger.error(msg)
            return TestRunResult(
                passed=False,
                output=msg,
                exit_code=127,
                command=cmd,
                python_bin=str(self.python_bin),
                used_fallback=self.used_fallback,
            )

    def _annotate_missing_deps(self, output: str, exit_code: int) -> str:
        """Append actionable guidance when imports fail (fail-safe messaging)."""
        if exit_code == 0:
            return output
        markers = ("ModuleNotFoundError", "ImportError", "No module named")
        if not any(m in output for m in markers):
            return output
        hint = (
            "\n\n[sdlc-orchestrator] Dependency import failed under "
            f"`{self.python_bin}` (fallback={self.used_fallback}).\n"
            "Fail-safe next steps:\n"
            f"  1. cd {self.repo_path} && python -m venv .venv\n"
            "  2. .venv/bin/pip install -r requirements.txt pytest\n"
            "  3. Or set TARGET_PYTHON_BIN to a Python that already has those packages.\n"
        )
        return output + hint

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
                python_bin=str(self.python_bin),
                used_fallback=self.used_fallback,
            )
        except subprocess.TimeoutExpired:
            return TestRunResult(
                passed=False,
                output=f"TIMEOUT after {self.timeout}s",
                exit_code=-1,
                command=command,
                python_bin=str(self.python_bin),
                used_fallback=self.used_fallback,
            )
