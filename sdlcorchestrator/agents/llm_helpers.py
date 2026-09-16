"""Shared helpers for agent LLM invocation and structured JSON parsing."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json_block(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response string."""
    text = text.strip()
    # Fenced block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    # Raw object
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object found in LLM response: {text[:200]}…")


def invoke_structured(
    llm: BaseChatModel,
    system: str,
    human: str,
    schema: type[T],
    fallback: T | None = None,
) -> T:
    """Invoke an LLM and parse the response into a Pydantic model.

    Prefers ``with_structured_output`` when available; falls back to JSON
    extraction from free-form text.
    """
    try:
        structured = llm.with_structured_output(schema)
        result = structured.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
    except Exception as exc:
        logger.debug("structured_output unavailable (%s); using JSON parse", exc)

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = " ".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )

    try:
        data = extract_json_block(str(content))
        return schema.model_validate(data)
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse LLM output into %s: %s", schema.__name__, exc)
        if fallback is not None:
            return fallback
        raise
