"""Helpers for validating API credentials and detecting placeholder values."""

from __future__ import annotations

_PLACEHOLDER_FRAGMENTS = (
    "sk-...",
    "ghp_...",
    "sk-ant-...",
    "aizasy_your_actual",
    "your_actual_gemini",
    "your-actual",
    "your-",
    "changeme",
    "replace",
    "xxx",
    "todo",
)


def is_usable_secret(value: str | None, *, min_length: int = 20) -> bool:
    """Return True when *value* looks like a real secret (not a .env placeholder).

    Accepts Google AI Studio keys in both ``AIza…`` and ``AQ.…`` forms.
    """
    if not value:
        return False
    stripped = value.strip()
    if len(stripped) < min_length:
        return False
    lower = stripped.lower()
    if any(frag in lower for frag in _PLACEHOLDER_FRAGMENTS):
        return False
    if stripped.endswith("..."):
        return False
    return True


def is_google_ai_studio_key(value: str | None) -> bool:
    """True for Google AI Studio API keys (``AIza…`` or ``AQ.…``)."""
    if not value:
        return False
    stripped = value.strip()
    return stripped.startswith(("AIza", "AQ.")) and is_usable_secret(stripped, min_length=20)
