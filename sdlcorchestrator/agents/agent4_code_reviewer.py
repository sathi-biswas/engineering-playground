"""Agent 4 — Senior Principal Code Reviewer."""

from __future__ import annotations

from typing import Any, Literal

from config import ModelTier, get_settings
from state import CodeReviewResult, SDLCState
from utils.git_helper import GitHelper, GitHelperError
from utils.logger import get_artifact_writer, get_logger
from utils.model_router import get_llm
from agents.llm_helpers import invoke_structured

logger = get_logger(__name__)

AGENT_NAME = "Agent4_CodeReviewer"

_SYSTEM = """You are a Principal Engineer performing a strict pull-request code review.
Evaluate: correctness, security, performance, maintainability, and unit-test coverage.
Decide APPROVED, REJECTED, or NEEDS_REVISION.
- APPROVED only if confidence >= 0.70 AND no critical issues.
- REJECTED if critical security/correctness flaws exist.
- NEEDS_REVISION if fixable issues remain.
Provide actionable inline_comments as {path, line, body} when possible.
"""


def run_code_reviewer(state: SDLCState) -> dict[str, Any]:
    """LangGraph node: review PR diff and post GitHub review comments."""
    writer = get_artifact_writer()
    settings = get_settings()
    writer.append_thought(AGENT_NAME, "Starting PR code review")

    updates: dict[str, Any] = {"current_agent": AGENT_NAME}

    try:
        pr_url = state.get("pr_url") or ""
        pr_number = state.get("pr_number")
        repo_path = state.get("target_repo_path") or "."
        github_repo = state.get("github_repo_name")

        git = GitHelper(repo_path=repo_path, github_repo_name=github_repo)

        diff_text = state.get("patch_diff") or ""
        if pr_number:
            try:
                diff_text = git.get_pr_diff(int(pr_number)) or diff_text
            except GitHelperError as exc:
                writer.append_thought(AGENT_NAME, f"Could not fetch PR diff: {exc}")
        if not diff_text:
            try:
                diff_text = git.get_diff()
            except GitHelperError:
                diff_text = "(no diff available)"

        llm = get_llm(tier=ModelTier.HIGH, temperature=0.1)
        human = (
            f"Bug ID: {state.get('bug_id')}\n"
            f"PR: {pr_url or pr_number or 'local diff'}\n\n"
            f"## Bug Description\n{state.get('bug_description', '')[:6000]}\n\n"
            f"## Analysis\n{state.get('code_analysis_summary', '')[:4000]}\n\n"
            f"## Test Results\npassed={ (state.get('test_results') or {}).get('passed') }\n"
            f"```\n{((state.get('test_results') or {}).get('output') or '')[:4000]}\n```\n\n"
            f"## Diff\n```diff\n{diff_text[:20000]}\n```\n"
        )

        fallback = CodeReviewResult(
            decision="NEEDS_REVISION",
            summary="Unable to complete automated review — insufficient context or LLM failure",
            inline_comments=[],
            critical_issues=["Review engine fallback triggered"],
            suggestions=["Re-run with valid GITHUB_TOKEN and LLM API key"],
            confidence=0.4,
        )
        result = invoke_structured(
            llm, _SYSTEM, human, CodeReviewResult, fallback=fallback
        )

        # Enforce policy: approve only if confidence >= threshold and no critical issues
        decision: Literal["APPROVED", "REJECTED", "NEEDS_REVISION"] = result.decision
        if decision == "APPROVED":
            if result.confidence < settings.confidence_threshold or result.critical_issues:
                decision = "NEEDS_REVISION"
                writer.append_thought(
                    AGENT_NAME,
                    "Downgraded APPROVED → NEEDS_REVISION due to confidence/critical issues",
                )

        # If tests never passed, reject
        if not (state.get("test_results") or {}).get("passed"):
            decision = "REJECTED"
            result.critical_issues = list(result.critical_issues) + [
                "Unit tests did not pass before review"
            ]

        updates["review_decision"] = decision
        updates["review_confidence"] = result.confidence
        updates["review_comments"] = _format_comments(result)
        updates["revision_count"] = int(state.get("revision_count") or 0) + (
            1 if decision == "NEEDS_REVISION" else 0
        )

        # Post to GitHub when PR exists
        if pr_number:
            event = {
                "APPROVED": "APPROVE",
                "REJECTED": "REQUEST_CHANGES",
                "NEEDS_REVISION": "REQUEST_CHANGES",
            }[decision]
            try:
                git.post_pr_review(
                    pr_number=int(pr_number),
                    body=_review_body(result, decision),
                    event=event,
                    comments=result.inline_comments,
                )
                writer.append_thought(AGENT_NAME, f"Posted GitHub review event={event}")
            except GitHelperError as exc:
                msg = f"Failed to post PR review: {exc}"
                writer.append_thought(AGENT_NAME, msg)
                updates.setdefault("error_logs", []).append(f"{AGENT_NAME}: {msg}")

        writer.write_markdown(
            filename="04_code_review_decision.md",
            title=f"Code Review Decision — {state.get('bug_id', 'N/A')}",
            sections={
                "Decision": decision,
                "Summary": result.summary,
                "Critical Issues": result.critical_issues or ["_None_"],
                "Suggestions": result.suggestions or ["_None_"],
                "Inline Comments": [
                    f"`{c.get('path')}:{c.get('line')}` — {c.get('body')}"
                    for c in result.inline_comments
                ]
                or ["_None_"],
                "Full Review Notes": updates["review_comments"],
                "PR URL": pr_url or "_N/A_",
            },
            metadata={
                "bug_id": state.get("bug_id"),
                "review_confidence": result.confidence,
                "threshold": settings.confidence_threshold,
                "pr_number": pr_number,
            },
        )

        updates["pipeline_status"] = (
            "SUCCESS" if decision == "APPROVED" else "RUNNING"
        )
        writer.append_thought(
            AGENT_NAME,
            f"Review complete decision={decision} confidence={result.confidence:.2f}",
        )
        return updates

    except Exception as exc:
        logger.exception("%s failed", AGENT_NAME)
        writer.write_error_artifact(AGENT_NAME, str(exc))
        return {
            **updates,
            "review_decision": "REJECTED",
            "review_confidence": 0.0,
            "review_comments": str(exc),
            "error_logs": [f"{AGENT_NAME}: {exc}"],
            "pipeline_status": "FAILED",
        }


def _format_comments(result: CodeReviewResult) -> str:
    parts = [result.summary, "", "### Critical Issues"]
    if result.critical_issues:
        parts.extend(f"- {i}" for i in result.critical_issues)
    else:
        parts.append("- None")
    parts.append("\n### Suggestions")
    if result.suggestions:
        parts.extend(f"- {s}" for s in result.suggestions)
    else:
        parts.append("- None")
    if result.inline_comments:
        parts.append("\n### Inline")
        for c in result.inline_comments:
            parts.append(f"- `{c.get('path')}:{c.get('line')}` — {c.get('body')}")
    parts.append(f"\n### Confidence\n{result.confidence}")
    return "\n".join(parts)


def _review_body(result: CodeReviewResult, decision: str) -> str:
    return (
        f"## Automated Review — **{decision}**\n\n"
        f"{result.summary}\n\n"
        f"**Confidence:** {result.confidence:.2f}\n\n"
        f"### Critical Issues\n"
        + ("\n".join(f"- {i}" for i in result.critical_issues) or "- None")
        + "\n\n### Suggestions\n"
        + ("\n".join(f"- {s}" for s in result.suggestions) or "- None")
        + "\n\n_Posted by SDLC Orchestrator Agent 4._"
    )
