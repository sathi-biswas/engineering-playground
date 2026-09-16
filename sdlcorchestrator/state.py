"""LangGraph agent state and Pydantic output schemas for the SDLC Orchestrator."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


def _merge_error_logs(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer that appends error log entries across graph nodes."""
    return (left or []) + (right or [])


class TestResults(TypedDict, total=False):
    """Structured unit-test execution outcome."""

    passed: bool
    output: str
    attempts: int
    exit_code: int


class SDLCState(TypedDict, total=False):
    """Shared mutable state flowing through the LangGraph SDLC pipeline."""

    # Inputs
    bug_id: str
    gdrive_file_id: str
    target_repo_path: str
    github_repo_name: str

    # Agent 1 — bug ingestion
    bug_description: str
    extraction_confidence: float

    # Agent 2 — code analysis
    affected_files: list[str]
    code_analysis_summary: str
    analysis_confidence: float

    # Agent 3 — patch & test
    patch_branch_name: str
    pr_url: str
    pr_number: int
    test_results: TestResults
    fix_confidence: float
    patch_diff: str
    fix_attempt: int

    # Agent 4 — code review
    review_comments: str
    review_decision: Literal["APPROVED", "REJECTED", "NEEDS_REVISION"]
    review_confidence: float
    revision_count: int

    # Cross-cutting
    error_logs: Annotated[list[str], _merge_error_logs]
    current_agent: str
    pipeline_status: Literal["RUNNING", "SUCCESS", "FAILED", "HALTED"]


# ---------------------------------------------------------------------------
# Pydantic schemas used by agents for structured LLM outputs
# ---------------------------------------------------------------------------


class BugExtraction(BaseModel):
    """Structured parse of a bug report document."""

    summary: str = Field(description="Concise Markdown summary of the bug")
    error_description: str = Field(description="Actionable error description")
    stack_traces: list[str] = Field(default_factory=list, description="Extracted stack traces")
    target_components: list[str] = Field(
        default_factory=list, description="Suspected modules/components"
    )
    reproduction_steps: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence 0-1")


class CodeAnalysisResult(BaseModel):
    """Structured codebase analysis correlating a bug to source locations."""

    architecture_overview: str = Field(description="High-level repo architecture notes")
    root_cause_hypothesis: str
    affected_files: list[str] = Field(description="Relative paths of files to modify")
    target_functions: list[str] = Field(
        default_factory=list, description="Functions/classes implicated"
    )
    suggested_approach: str = Field(description="How to fix the issue")
    confidence: float = Field(ge=0.0, le=1.0)


class PatchPlan(BaseModel):
    """Proposed code changes and accompanying unit tests."""

    branch_name: str
    file_edits: list[dict[str, Any]] = Field(
        description="List of {path, action, content_or_diff} edits"
    )
    unit_test_files: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {path, content} for new/updated tests",
    )
    commit_message: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class TestEvaluation(BaseModel):
    """LLM evaluation of raw pytest output."""

    passed: bool
    failure_summary: str = ""
    suggested_fixes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class CodeReviewResult(BaseModel):
    """Structured PR review decision."""

    decision: Literal["APPROVED", "REJECTED", "NEEDS_REVISION"]
    summary: str
    inline_comments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional {path, line, body} review comments",
    )
    critical_issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
