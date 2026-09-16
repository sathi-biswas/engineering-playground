"""Thought-process logger and Markdown artifact writer for `/output`."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import get_settings

# ---------------------------------------------------------------------------
# Standard Python logging setup
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once for CLI / FastAPI entrypoints."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; ensures logging is configured."""
    setup_logging()
    return logging.getLogger(name)


class ArtifactWriter:
    """Writes structured Markdown artifacts and agent thought logs to `/output`."""

    def __init__(self, output_dir: Path | None = None) -> None:
        settings = get_settings()
        self.output_dir = Path(output_dir or settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("artifact_writer")

    def write_markdown(
        self,
        filename: str,
        title: str,
        sections: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write a Markdown artifact and return its path.

        Args:
            filename: Target file name under `/output` (e.g. ``01_bug_description.md``).
            title: H1 title for the document.
            sections: Ordered mapping of section heading -> body (str or list).
            metadata: Optional key/value front-matter rendered under the title.
        """
        path = self.output_dir / filename
        lines: list[str] = [
            f"# {title}",
            "",
            f"_Generated at {datetime.now(timezone.utc).isoformat()}_",
            "",
        ]

        if metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in metadata.items():
                lines.append(f"- **{key}**: `{value}`")
            lines.append("")

        for heading, body in sections.items():
            lines.append(f"## {heading}")
            lines.append("")
            if isinstance(body, list):
                for item in body:
                    lines.append(f"- {item}")
            elif isinstance(body, dict):
                for k, v in body.items():
                    lines.append(f"- **{k}**: {v}")
            else:
                lines.append(str(body))
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        self._logger.info("Wrote artifact: %s", path)
        return path

    def write_error_artifact(self, agent_name: str, error: str, context: str = "") -> Path:
        """Write a fallback error Markdown file when a node fails."""
        safe_name = agent_name.replace(" ", "_").lower()
        filename = f"error_{safe_name}.md"
        return self.write_markdown(
            filename=filename,
            title=f"Error — {agent_name}",
            sections={
                "Error": f"```\n{error}\n```",
                "Context": context or "_No additional context._",
            },
            metadata={"agent": agent_name, "status": "FAILED"},
        )

    def append_thought(self, agent_name: str, thought: str) -> Path:
        """Append an agent thought-process entry to a running log file."""
        path = self.output_dir / "thought_process.log"
        stamp = datetime.now(timezone.utc).isoformat()
        entry = f"[{stamp}] [{agent_name}] {thought}\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(entry)
        self._logger.debug("Thought logged for %s", agent_name)
        return path


# Module-level convenience instance (lazy via factory)
_writer: ArtifactWriter | None = None


def get_artifact_writer() -> ArtifactWriter:
    """Return a shared ArtifactWriter instance."""
    global _writer
    if _writer is None:
        _writer = ArtifactWriter()
    return _writer
