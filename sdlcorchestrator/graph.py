"""LangGraph orchestration: stateful SDLC pipeline with conditional routing."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agents.agent1_bug_reader import run_bug_reader
from agents.agent2_code_analyzer import run_code_analyzer
from agents.agent3_bug_fixer import run_bug_fixer
from agents.agent4_code_reviewer import run_code_reviewer
from config import get_settings
from state import SDLCState
from utils.logger import get_artifact_writer, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def error_handler_node(state: SDLCState) -> dict[str, Any]:
    """Terminal / halt node when confidence gates or hard failures trip."""
    writer = get_artifact_writer()
    errors = state.get("error_logs") or []
    reason_parts = []

    analysis_conf = float(state.get("analysis_confidence") or 0.0)
    settings = get_settings()
    if analysis_conf < settings.confidence_threshold and state.get("current_agent") in {
        "Agent2_CodeAnalyzer",
        "error_handler",
    }:
        reason_parts.append(
            f"analysis_confidence {analysis_conf:.2f} < {settings.confidence_threshold}"
        )

    tests = state.get("test_results") or {}
    if tests.get("attempts") and not tests.get("passed"):
        reason_parts.append(
            f"tests failed after {tests.get('attempts', '?')} attempt(s)"
        )

    if state.get("review_decision") == "REJECTED":
        reason_parts.append("code review REJECTED")

    if not reason_parts and errors:
        reason_parts.append(errors[-1])
    if not reason_parts:
        reason_parts.append("Pipeline halted for unknown reason")

    reason = "; ".join(reason_parts)
    writer.write_markdown(
        filename="00_pipeline_halted.md",
        title="Pipeline Halted",
        sections={
            "Reason": reason,
            "Error Logs": errors or ["_None_"],
            "Last Agent": state.get("current_agent", "unknown"),
            "State Snapshot": {
                "bug_id": state.get("bug_id"),
                "analysis_confidence": state.get("analysis_confidence"),
                "fix_confidence": state.get("fix_confidence"),
                "fix_attempt": state.get("fix_attempt"),
                "review_decision": state.get("review_decision"),
                "pr_url": state.get("pr_url"),
            },
        },
        metadata={"pipeline_status": "HALTED"},
    )
    writer.append_thought("error_handler", reason)
    logger.error("Pipeline halted: %s", reason)
    return {
        "current_agent": "error_handler",
        "pipeline_status": "HALTED",
        "error_logs": [f"error_handler: {reason}"],
    }


def finalize_node(state: SDLCState) -> dict[str, Any]:
    """Emit a final summary artifact when the pipeline completes successfully."""
    writer = get_artifact_writer()
    decision = state.get("review_decision", "N/A")
    status = "SUCCESS" if decision == "APPROVED" else state.get("pipeline_status", "RUNNING")
    writer.write_markdown(
        filename="05_pipeline_summary.md",
        title="SDLC Pipeline Summary",
        sections={
            "Status": status,
            "Review Decision": decision,
            "PR URL": state.get("pr_url") or "_N/A_",
            "Affected Files": state.get("affected_files") or [],
            "Confidences": {
                "extraction": state.get("extraction_confidence"),
                "analysis": state.get("analysis_confidence"),
                "fix": state.get("fix_confidence"),
                "review": state.get("review_confidence"),
            },
            "Errors": state.get("error_logs") or ["_None_"],
        },
        metadata={"bug_id": state.get("bug_id")},
    )
    return {"pipeline_status": status if status != "RUNNING" else "SUCCESS"}


# ---------------------------------------------------------------------------
# Conditional routers
# ---------------------------------------------------------------------------


def route_after_analysis(
    state: SDLCState,
) -> Literal["agent3_bug_fixer", "error_handler"]:
    """Gate: require analysis_confidence >= threshold before patching."""
    settings = get_settings()
    conf = float(state.get("analysis_confidence") or 0.0)
    if conf < settings.confidence_threshold:
        logger.warning(
            "Routing to error_handler — analysis_confidence=%.2f < %.2f",
            conf,
            settings.confidence_threshold,
        )
        return "error_handler"
    return "agent3_bug_fixer"


def route_after_fix(
    state: SDLCState,
) -> Literal["agent3_bug_fixer", "agent4_code_reviewer", "error_handler"]:
    """Retry failed tests up to max_fix_retries, else continue or halt."""
    settings = get_settings()
    tests = state.get("test_results") or {}
    attempt = int(state.get("fix_attempt") or 0)
    passed = bool(tests.get("passed"))
    conf = float(state.get("fix_confidence") or 0.0)

    if passed and conf >= settings.confidence_threshold and state.get("pr_url"):
        return "agent4_code_reviewer"

    if passed and conf >= settings.confidence_threshold:
        # Local success without PR — still allow review on local diff
        return "agent4_code_reviewer"

    if not passed and attempt < settings.max_fix_retries:
        logger.info(
            "Retrying Agent 3 (attempt %s/%s)", attempt, settings.max_fix_retries
        )
        return "agent3_bug_fixer"

    if not passed:
        return "error_handler"

    # Passed but low confidence — still review locally
    return "agent4_code_reviewer"


def route_after_review(
    state: SDLCState,
) -> Literal["agent3_bug_fixer", "finalize", "error_handler"]:
    """Optionally loop back to Agent 3 on NEEDS_REVISION."""
    settings = get_settings()
    decision = state.get("review_decision")
    revision_count = int(state.get("revision_count") or 0)

    if decision == "APPROVED":
        return "finalize"

    if (
        decision == "NEEDS_REVISION"
        and settings.enable_revision_loop
        and revision_count <= settings.max_fix_retries
    ):
        logger.info("Revision requested — routing back to Agent 3")
        return "agent3_bug_fixer"

    if decision == "REJECTED":
        return "error_handler"

    return "finalize"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """Compile the SDLC StateGraph and return a runnable app."""
    graph = StateGraph(SDLCState)

    graph.add_node("agent1_bug_reader", run_bug_reader)
    graph.add_node("agent2_code_analyzer", run_code_analyzer)
    graph.add_node("agent3_bug_fixer", run_bug_fixer)
    graph.add_node("agent4_code_reviewer", run_code_reviewer)
    graph.add_node("error_handler", error_handler_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("agent1_bug_reader")
    graph.add_edge("agent1_bug_reader", "agent2_code_analyzer")
    graph.add_conditional_edges(
        "agent2_code_analyzer",
        route_after_analysis,
        {
            "agent3_bug_fixer": "agent3_bug_fixer",
            "error_handler": "error_handler",
        },
    )
    graph.add_conditional_edges(
        "agent3_bug_fixer",
        route_after_fix,
        {
            "agent3_bug_fixer": "agent3_bug_fixer",
            "agent4_code_reviewer": "agent4_code_reviewer",
            "error_handler": "error_handler",
        },
    )
    graph.add_conditional_edges(
        "agent4_code_reviewer",
        route_after_review,
        {
            "agent3_bug_fixer": "agent3_bug_fixer",
            "finalize": "finalize",
            "error_handler": "error_handler",
        },
    )
    graph.add_edge("error_handler", END)
    graph.add_edge("finalize", END)

    app = graph.compile()
    logger.info("SDLC LangGraph compiled successfully")
    return app


def run_pipeline(initial_state: SDLCState) -> SDLCState:
    """Execute the full pipeline synchronously and return the final state."""
    app = build_graph()
    writer = get_artifact_writer()
    writer.append_thought(
        "orchestrator",
        f"Pipeline start bug_id={initial_state.get('bug_id')} "
        f"repo={initial_state.get('github_repo_name')}",
    )

    # Ensure mutable defaults
    seed: SDLCState = {
        "error_logs": [],
        "affected_files": [],
        "fix_attempt": 0,
        "revision_count": 0,
        "pipeline_status": "RUNNING",
        "test_results": {"passed": False, "output": "", "attempts": 0},
        **initial_state,
    }

    final: SDLCState = app.invoke(seed)
    logger.info(
        "Pipeline finished status=%s decision=%s pr=%s",
        final.get("pipeline_status"),
        final.get("review_decision"),
        final.get("pr_url"),
    )
    return final
