"""Runtime configuration for CodeCracker."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Pipeline knobs — override via env or CLI."""

    work_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / ".repos"
    )
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "output"
    )
    max_files: int = 400
    max_file_bytes: int = 200_000
    max_tour_steps: int = 25
    shallow_clone: bool = True

    # LLM — set OPENAI_API_KEY or ANTHROPIC_API_KEY; otherwise heuristic mode
    llm_provider: str = field(
        default_factory=lambda: os.getenv("CODECRACKER_LLM", "auto")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("CODECRACKER_OPENAI_MODEL", "gpt-4o-mini")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv(
            "CODECRACKER_ANTHROPIC_MODEL", "claude-3-5-haiku-latest"
        )
    )
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )

    def ensure_dirs(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
