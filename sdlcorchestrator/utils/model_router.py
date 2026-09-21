"""Dynamic LLM factory with FinOps model-tier routing.

IMPORTANT — Gemini authentication:
  Always use ``ChatGoogleGenerativeAI`` from ``langchain-google-genai``.
  That SDK calls Google's native Generative Language REST API and accepts
  Google AI Studio keys (including ``AQ.``-prefixed keys) via ``google_api_key``.

  Do NOT route Gemini through OpenAI-compatible wrappers, e.g.::

      ChatOpenAI(
          api_key=gemini_key,
          base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
      )

  Bearer auth against that OpenAI-compat endpoint returns 400/401 for AQ. keys.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from config import ModelTier, get_settings
from utils.logger import get_logger
from utils.secrets import is_usable_secret

logger = get_logger(__name__)

# Hard-reject accidental OpenAI-compat Gemini routing
_FORBIDDEN_GEMINI_OPENAI_BASE = "generativelanguage.googleapis.com"


def get_llm(
    tier: ModelTier | str = ModelTier.MID,
    temperature: float = 0.2,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a chat model for the requested FinOps tier.

    Selects Gemini (default), OpenAI, or Anthropic based on ``Settings.llm_provider``.
    Falls back to a lightweight stub when no API key is configured so the
    graph can still be exercised in dry-run / CI scaffolding mode.
    """
    settings = get_settings()
    tier_enum = ModelTier(tier) if isinstance(tier, str) else tier
    model_name = settings.model_for_tier(tier_enum)

    if settings.llm_provider == "gemini":
        return _build_gemini_llm(model_name, tier_enum, temperature, **kwargs)

    if settings.llm_provider == "anthropic":
        key = (
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key
            else None
        )
        if not is_usable_secret(key):
            logger.warning(
                "ANTHROPIC_API_KEY missing/placeholder — using StubLLM for tier=%s", tier_enum
            )
            return StubLLM(model_name=model_name, tier=tier_enum)
        from langchain_anthropic import ChatAnthropic

        logger.info("Routing to Anthropic model=%s tier=%s", model_name, tier_enum.value)
        return ChatAnthropic(
            model=model_name,
            api_key=key,
            temperature=temperature,
            **kwargs,
        )

    # OpenAI provider only — never used for Gemini / AQ. keys
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    if not is_usable_secret(key):
        logger.warning("OPENAI_API_KEY missing/placeholder — using StubLLM for tier=%s", tier_enum)
        return StubLLM(model_name=model_name, tier=tier_enum)

    base_url = str(kwargs.get("base_url") or "")
    if _FORBIDDEN_GEMINI_OPENAI_BASE in base_url:
        raise ValueError(
            "Refusing ChatOpenAI against Google's OpenAI-compat endpoint. "
            "Use LLM_PROVIDER=gemini with ChatGoogleGenerativeAI instead."
        )

    from langchain_openai import ChatOpenAI

    # o-series models reject custom temperature in some API versions
    call_kwargs: dict[str, Any] = dict(kwargs)
    if model_name.startswith("o"):
        call_kwargs.pop("temperature", None)
    else:
        call_kwargs["temperature"] = temperature

    logger.info("Routing to OpenAI model=%s tier=%s", model_name, tier_enum.value)
    return ChatOpenAI(model=model_name, api_key=key, **call_kwargs)


def _build_gemini_llm(
    model_name: str,
    tier_enum: ModelTier,
    temperature: float,
    **kwargs: Any,
) -> BaseChatModel:
    """Build a native Gemini chat model (never ChatOpenAI / OpenAI-compat)."""
    settings = get_settings()
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
    if not is_usable_secret(key, min_length=20):
        logger.warning(
            "GEMINI_API_KEY missing/placeholder — using StubLLM for tier=%s", tier_enum
        )
        return StubLLM(model_name=model_name, tier=tier_enum)

    # Drop any OpenAI-style kwargs that would imply compat wrapping
    safe_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in {"base_url", "openai_api_base", "api_key", "openai_api_key"}
    }

    from langchain_google_genai import ChatGoogleGenerativeAI

    logger.info(
        "Routing to Gemini via ChatGoogleGenerativeAI model=%s tier=%s (native REST, not OpenAI-compat)",
        model_name,
        tier_enum.value,
    )
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=temperature,
        **safe_kwargs,
    )


class StubLLM(BaseChatModel):
    """Deterministic stand-in used when API keys are absent (local dry-run)."""

    model_name: str = "stub"
    tier: ModelTier = ModelTier.MID

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages: list, stop: list[str] | None = None, **kwargs: Any):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        content = (
            f"[StubLLM tier={self.tier.value} model={self.model_name}] "
            "No API key configured. Returning placeholder structured response.\n"
            '{"summary":"Stub bug summary","error_description":"Stub error",'
            '"stack_traces":[],"target_components":["unknown"],'
            '"reproduction_steps":[],"confidence":0.5,'
            '"architecture_overview":"Stub architecture",'
            '"root_cause_hypothesis":"Insufficient context without LLM",'
            '"affected_files":[],"target_functions":[],'
            '"suggested_approach":"Configure API keys and re-run",'
            '"branch_name":"fix/bug-stub","file_edits":[],'
            '"unit_test_files":[],"commit_message":"chore: stub",'
            '"rationale":"stub","passed":false,"failure_summary":"stub",'
            '"suggested_fixes":[],"decision":"NEEDS_REVISION",'
            '"inline_comments":[],"critical_issues":["No live LLM"],'
            '"suggestions":["Set GEMINI_API_KEY (preferred), OPENAI_API_KEY, or ANTHROPIC_API_KEY"]}'
        )
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    async def _agenerate(self, messages: list, stop: list[str] | None = None, **kwargs: Any):
        return self._generate(messages, stop=stop, **kwargs)
