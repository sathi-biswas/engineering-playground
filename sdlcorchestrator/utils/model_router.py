"""Dynamic LLM factory with FinOps model-tier routing."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from config import ModelTier, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def get_llm(
    tier: ModelTier | str = ModelTier.MID,
    temperature: float = 0.2,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a chat model for the requested FinOps tier.

    Selects OpenAI or Anthropic based on ``Settings.llm_provider``.
    Falls back to a lightweight stub when no API key is configured so the
    graph can still be exercised in dry-run / CI scaffolding mode.
    """
    settings = get_settings()
    tier_enum = ModelTier(tier) if isinstance(tier, str) else tier
    model_name = settings.model_for_tier(tier_enum)

    if settings.llm_provider == "anthropic":
        key = (
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key
            else None
        )
        if not key:
            logger.warning("ANTHROPIC_API_KEY missing — using StubLLM for tier=%s", tier_enum)
            return StubLLM(model_name=model_name, tier=tier_enum)
        from langchain_anthropic import ChatAnthropic

        logger.info("Routing to Anthropic model=%s tier=%s", model_name, tier_enum.value)
        return ChatAnthropic(
            model=model_name,
            api_key=key,
            temperature=temperature,
            **kwargs,
        )

    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    if not key:
        logger.warning("OPENAI_API_KEY missing — using StubLLM for tier=%s", tier_enum)
        return StubLLM(model_name=model_name, tier=tier_enum)

    from langchain_openai import ChatOpenAI

    # o-series models reject custom temperature in some API versions
    call_kwargs: dict[str, Any] = dict(kwargs)
    if model_name.startswith("o"):
        call_kwargs.pop("temperature", None)
    else:
        call_kwargs["temperature"] = temperature

    logger.info("Routing to OpenAI model=%s tier=%s", model_name, tier_enum.value)
    return ChatOpenAI(model=model_name, api_key=key, **call_kwargs)


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
            '"suggestions":["Set OPENAI_API_KEY or ANTHROPIC_API_KEY"]}'
        )
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    async def _agenerate(self, messages: list, stop: list[str] | None = None, **kwargs: Any):
        return self._generate(messages, stop=stop, **kwargs)
