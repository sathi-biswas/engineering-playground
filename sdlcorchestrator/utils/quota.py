"""Gemini / LLM quota detection and fail-fast helpers."""

from __future__ import annotations

from typing import Any


class QuotaExceededError(RuntimeError):
    """Raised when the LLM provider reports rate/quota exhaustion (HTTP 429)."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


def is_quota_error(exc: BaseException | None) -> bool:
    """Return True when *exc* looks like a free-tier / rate-limit exhaustion."""
    if exc is None:
        return False
    if isinstance(exc, QuotaExceededError):
        return True

    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = (
        "resourceexhausted",
        "resource exhausted",
        "quota exceeded",
        "exceeded your current quota",
        "rate limit",
        "ratelimit",
        "too many requests",
        "generate_content_free_tier",
        "429",
    )
    if "resourceexhausted" in name or "toomanyrequests" in name:
        return True
    return any(m in text for m in markers)


def raise_if_quota_error(exc: BaseException) -> None:
    """Re-raise *exc* as ``QuotaExceededError`` when it is a quota/429 failure."""
    if is_quota_error(exc):
        raise QuotaExceededError(
            "LLM quota exceeded (HTTP 429). Wait for daily reset (midnight PT), "
            "switch MODEL_* to a flash-lite model (~500 RPD), or enable billing. "
            f"Original: {exc}",
            cause=exc,
        ) from exc


def quota_error_state_update(agent_name: str, exc: BaseException) -> dict[str, Any]:
    """Build a LangGraph state patch that forces the confidence gate to halt."""
    msg = f"{agent_name}: QUOTA_EXCEEDED — {exc}"
    return {
        "current_agent": agent_name,
        "extraction_confidence": 0.0,
        "analysis_confidence": 0.0,
        "fix_confidence": 0.0,
        "pipeline_status": "HALTED",
        "error_logs": [msg],
        "bug_description": f"ERROR: {exc}",
        "code_analysis_summary": f"ERROR: {exc}",
    }
