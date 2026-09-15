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
    # Partial clone (blob:none) — less bandwidth on large public repos
    clone_filter_blob_none: bool = True
    # Minimum seconds between *fresh* clones (0 disables). Reuses are free.
    min_clone_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("CODECRACKER_CLONE_INTERVAL", "5"))
    )
    # Optional PAT — authenticated git HTTPS raises GitHub rate limits substantially
    github_token: str | None = field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
        or os.getenv("CODECRACKER_GITHUB_TOKEN")
    )

    # Token budgeting (tiktoken) — trim structural context before LLM serialize
    llm_context_token_budget: int = field(
        default_factory=lambda: int(
            os.getenv("CODECRACKER_CONTEXT_TOKEN_BUDGET", "6000")
        )
    )
    # Reserve tokens for system prompt + model JSON reply inside the same call.
    # When > 0 and less than llm_context_token_budget, usable context =
    # budget − reserve (treat budget as a combined window).
    llm_output_token_reserve: int = field(
        default_factory=lambda: int(
            os.getenv("CODECRACKER_OUTPUT_TOKEN_RESERVE", "0")
        )
    )
    tiktoken_encoding: str = field(
        default_factory=lambda: os.getenv(
            "CODECRACKER_TIKTOKEN_ENCODING", "cl100k_base"
        )
    )

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
