"""Agent 1 — Bug Ingestion via Google Drive RAG."""

from __future__ import annotations

from typing import Any

from config import ModelTier, get_settings
from state import BugExtraction, SDLCState
from tools.gdrive_rag import GoogleDriveRAG, GoogleDriveRAGError
from utils.logger import get_artifact_writer, get_logger
from utils.model_router import get_llm
from agents.llm_helpers import invoke_structured

logger = get_logger(__name__)

AGENT_NAME = "Agent1_BugReader"

_SYSTEM = """You are a Bug Ingestion Analyst for an automated SDLC pipeline.
Parse the provided bug report into structured fields.
Extract actionable error descriptions, stack traces, and likely target components.
Respond with high-quality structured data. Confidence must reflect extraction quality (0-1).
"""


def run_bug_reader(state: SDLCState) -> dict[str, Any]:
    """LangGraph node: ingest bug report from Google Drive (or local path)."""
    writer = get_artifact_writer()
    settings = get_settings()
    writer.append_thought(AGENT_NAME, "Starting bug ingestion")

    updates: dict[str, Any] = {"current_agent": AGENT_NAME, "pipeline_status": "RUNNING"}

    try:
        file_id = state.get("gdrive_file_id") or ""
        if not file_id:
            raise GoogleDriveRAGError("gdrive_file_id is required")

        rag = GoogleDriveRAG()
        raw_text, rag_chunks = rag.ingest_bug_report(file_id)
        writer.append_thought(
            AGENT_NAME, f"Loaded {len(raw_text)} chars; indexed {len(rag_chunks)} chunks"
        )

        # Mid tier if report is large/complex; otherwise low
        tier = ModelTier.MID if len(raw_text) > 4000 else ModelTier.LOW
        llm = get_llm(tier=tier, temperature=0.1)

        human = (
            f"Bug ID: {state.get('bug_id', 'unknown')}\n\n"
            f"## Raw Bug Report\n{raw_text[:20000]}\n\n"
            f"## RAG Context Chunks\n" + "\n---\n".join(rag_chunks[:5])
        )

        fallback = BugExtraction(
            summary="Failed to parse bug report",
            error_description=raw_text[:2000],
            stack_traces=[],
            target_components=[],
            reproduction_steps=[],
            confidence=0.4,
        )
        extraction = invoke_structured(llm, _SYSTEM, human, BugExtraction, fallback=fallback)

        bug_md = _format_bug_markdown(raw_text, extraction, rag_chunks)
        updates["bug_description"] = bug_md
        updates["extraction_confidence"] = extraction.confidence

        writer.write_markdown(
            filename="01_bug_description.md",
            title=f"Bug Description — {state.get('bug_id', 'N/A')}",
            sections={
                "Summary": extraction.summary,
                "Error Description": extraction.error_description,
                "Stack Traces": extraction.stack_traces or ["_None extracted_"],
                "Target Components": extraction.target_components or ["_Unknown_"],
                "Reproduction Steps": extraction.reproduction_steps or ["_Not provided_"],
                "Raw Bug Report": f"```\n{raw_text[:15000]}\n```",
                "RAG Chunks": rag_chunks[:5] or ["_No chunks_"],
            },
            metadata={
                "bug_id": state.get("bug_id"),
                "gdrive_file_id": file_id,
                "extraction_confidence": extraction.confidence,
                "model_tier": tier.value,
                "threshold": settings.confidence_threshold,
            },
        )

        if extraction.confidence < settings.confidence_threshold:
            msg = (
                f"Extraction confidence {extraction.confidence:.2f} "
                f"< threshold {settings.confidence_threshold}"
            )
            writer.append_thought(AGENT_NAME, msg)
            updates.setdefault("error_logs", []).append(f"{AGENT_NAME}: {msg}")
            # Continue pipeline — Agent 2 may still succeed with raw text
        else:
            writer.append_thought(
                AGENT_NAME, f"Success — confidence={extraction.confidence:.2f}"
            )

        return updates

    except Exception as exc:
        logger.exception("%s failed", AGENT_NAME)
        writer.write_error_artifact(AGENT_NAME, str(exc), context=str(state.get("bug_id")))
        return {
            **updates,
            "bug_description": state.get("bug_description") or f"ERROR: {exc}",
            "extraction_confidence": 0.0,
            "error_logs": [f"{AGENT_NAME}: {exc}"],
            "pipeline_status": "RUNNING",  # let graph decide halt later
        }


def _format_bug_markdown(raw: str, extraction: BugExtraction, chunks: list[str]) -> str:
    """Compose the canonical bug_description field stored in state."""
    parts = [
        f"## Summary\n{extraction.summary}",
        f"## Error\n{extraction.error_description}",
    ]
    if extraction.stack_traces:
        parts.append("## Stack Traces\n" + "\n\n".join(f"```\n{s}\n```" for s in extraction.stack_traces))
    if extraction.target_components:
        parts.append("## Components\n" + ", ".join(extraction.target_components))
    if extraction.reproduction_steps:
        parts.append(
            "## Reproduction\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(extraction.reproduction_steps, 1))
        )
    parts.append(f"## Confidence\n{extraction.confidence}")
    return "\n\n".join(parts)
