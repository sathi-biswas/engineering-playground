"""Centralized settings for the SDLC Orchestrator.

Loads configuration from environment variables and optional `.env` file.
Model tiers implement FinOps routing: cheaper models for parse/eval work,
stronger models for architecture and review.

Default provider is Google Gemini (free tier via Google AI Studio).
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelTier(str, Enum):
    """FinOps model routing tiers."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
    output_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "output")
    vector_store_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / ".vector_store"
    )

    # --- LLM providers ---
    # Default: Gemini (free from Google AI Studio)
    llm_provider: Literal["gemini", "openai", "anthropic"] = Field(
        default="gemini", alias="LLM_PROVIDER"
    )
    gemini_api_key: Optional[SecretStr] = Field(default=None, alias="GEMINI_API_KEY")
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[SecretStr] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # --- Model tiers (FinOps) — Gemini free-tier defaults ---
    # Prefer flash-lite (~500 RPD) over full flash (~20 RPD) for local iteration.
    # Low: parse, format, test-result evaluation
    model_low: str = Field(default="gemini-3.5-flash-lite", alias="MODEL_LOW")
    # Mid: RAG correlation, structure analysis, patch generation
    model_mid: str = Field(default="gemini-3.5-flash-lite", alias="MODEL_MID")
    # High: final verification, edge-case tests, strict PR review
    model_high: str = Field(default="gemini-3.5-flash-lite", alias="MODEL_HIGH")

    # OpenAI alternatives when LLM_PROVIDER=openai
    model_low_openai: str = Field(default="gpt-4o-mini", alias="MODEL_LOW_OPENAI")
    model_mid_openai: str = Field(default="gpt-4o", alias="MODEL_MID_OPENAI")
    model_high_openai: str = Field(default="o3-mini", alias="MODEL_HIGH_OPENAI")

    # Anthropic alternatives when LLM_PROVIDER=anthropic
    model_low_anthropic: str = Field(default="claude-3-haiku-20240307", alias="MODEL_LOW_ANTHROPIC")
    model_mid_anthropic: str = Field(
        default="claude-3-5-sonnet-20241022", alias="MODEL_MID_ANTHROPIC"
    )
    model_high_anthropic: str = Field(
        default="claude-3-5-sonnet-20241022", alias="MODEL_HIGH_ANTHROPIC"
    )

    # --- Confidence & retry thresholds ---
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    max_fix_retries: int = Field(default=2, ge=0, le=5)
    enable_revision_loop: bool = Field(default=True, alias="ENABLE_REVISION_LOOP")

    # --- GitHub ---
    github_token: Optional[SecretStr] = Field(default=None, alias="GITHUB_TOKEN")
    github_default_base_branch: str = Field(default="main", alias="GITHUB_BASE_BRANCH")

    # --- Google Drive ---
    google_application_credentials: Optional[Path] = Field(
        default=None, alias="GOOGLE_APPLICATION_CREDENTIALS"
    )
    gdrive_folder_id: Optional[str] = Field(default=None, alias="GDRIVE_FOLDER_ID")
    gdrive_bug_file_id: Optional[str] = Field(default=None, alias="GDRIVE_BUG_FILE_ID")

    # --- Runtime defaults (overridable via CLI) ---
    default_target_repo_path: Optional[Path] = Field(default=None, alias="TARGET_REPO_PATH")
    default_github_repo_name: Optional[str] = Field(default=None, alias="GITHUB_REPO_NAME")

    # --- Test runner ---
    test_timeout_seconds: int = Field(default=300, ge=30, le=3600)
    pytest_args: str = Field(default="-q --tb=short", alias="PYTEST_ARGS")
    # Prefer the target repo's venv so its deps (rich, jinja2, …) are available.
    # Leave unset to auto-detect `.venv` / `venv` / `codecracker_venv` under the repo.
    target_python_bin: Optional[str] = Field(default=None, alias="TARGET_PYTHON_BIN")

    def model_for_tier(self, tier: Union[ModelTier, str]) -> str:
        """Resolve the concrete model name for a FinOps tier."""
        tier_enum = ModelTier(tier) if isinstance(tier, str) else tier
        if self.llm_provider == "anthropic":
            mapping = {
                ModelTier.LOW: self.model_low_anthropic,
                ModelTier.MID: self.model_mid_anthropic,
                ModelTier.HIGH: self.model_high_anthropic,
            }
        elif self.llm_provider == "openai":
            mapping = {
                ModelTier.LOW: self.model_low_openai,
                ModelTier.MID: self.model_mid_openai,
                ModelTier.HIGH: self.model_high_openai,
            }
        else:
            # gemini (default) — MODEL_LOW / MID / HIGH apply directly
            mapping = {
                ModelTier.LOW: self.model_low,
                ModelTier.MID: self.model_mid,
                ModelTier.HIGH: self.model_high,
            }
        return mapping[tier_enum]

    def ensure_directories(self) -> None:
        """Create output and vector-store directories if missing."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings
