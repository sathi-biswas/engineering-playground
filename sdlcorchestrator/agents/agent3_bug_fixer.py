"""Agent 3 — Senior Developer Patch, Test, and PR Agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from config import ModelTier, get_settings
from state import PatchPlan, SDLCState, TestEvaluation, TestResults
from tools.code_parser import CodeParser
from utils.git_helper import GitHelper, GitHelperError
from utils.logger import get_artifact_writer, get_logger
from utils.model_router import get_llm
from utils.test_runner import TestRunner
from utils.quota import QuotaExceededError
from agents.llm_helpers import invoke_structured

logger = get_logger(__name__)

AGENT_NAME = "Agent3_BugFixer"

_PATCH_SYSTEM = """You are a Senior Software Engineer fixing a production bug.
Produce a concrete patch plan:
- file_edits: list of objects with keys path, action ("write"|"replace"), content (full file for write)
  Prefer action="write" with the COMPLETE updated file content.
- unit_test_files: list of {path, content} for new or updated pytest tests
- branch_name: fix/bug-<id>
- commit_message: conventional commit style
- confidence: 0-1
Keep changes minimal and focused on the root cause. Do not invent unrelated refactors.
"""

_TEST_EVAL_SYSTEM = """You are a test-result analyst. Given pytest output, decide if tests passed,
summarize failures, and suggest concrete code fixes. Keep confidence honest (0-1).
"""


def run_bug_fixer(state: SDLCState) -> dict[str, Any]:
    """LangGraph node: generate patch, run tests (with retries), open PR."""
    writer = get_artifact_writer()
    settings = get_settings()
    attempt = int(state.get("fix_attempt") or 0) + 1
    writer.append_thought(AGENT_NAME, f"Starting patch attempt {attempt}")

    updates: dict[str, Any] = {
        "current_agent": AGENT_NAME,
        "fix_attempt": attempt,
    }

    prior_logs = state.get("error_logs") or []
    if any(str(e).startswith("QUOTA_EXCEEDED") for e in prior_logs):
        msg = "Skipping patch — prior agent hit LLM quota"
        writer.append_thought(AGENT_NAME, msg)
        writer.write_error_artifact(AGENT_NAME, msg, context="QUOTA_EXCEEDED")
        return {
            **updates,
            "fix_confidence": 0.0,
            "test_results": {
                "passed": False,
                "output": msg,
                "attempts": attempt,
                "exit_code": -1,
            },
            "error_logs": [f"QUOTA_EXCEEDED: {AGENT_NAME}: {msg}"],
            "pipeline_status": "HALTED",
        }

    try:
        bug_id = state.get("bug_id") or "unknown"
        repo_path = Path(state.get("target_repo_path") or ".")
        branch_name = state.get("patch_branch_name") or f"fix/bug-{_slug(bug_id)}"

        parser = CodeParser(repo_path)
        affected = state.get("affected_files") or []
        context = parser.gather_context(affected, max_files=12)
        context_blob = "\n\n".join(
            f"### `{p}`\n```python\n{src}\n```" for p, src in context.items()
        )

        prior_feedback = ""
        if state.get("review_comments") and state.get("review_decision") == "NEEDS_REVISION":
            prior_feedback = f"\n## Reviewer Feedback to Address\n{state['review_comments']}\n"
        test_feedback = ""
        prev_tests = state.get("test_results") or {}
        if attempt > 1 and not prev_tests.get("passed"):
            test_feedback = (
                f"\n## Previous Test Failure (attempt {attempt - 1})\n"
                f"```\n{(prev_tests.get('output') or '')[:6000]}\n```\n"
                "Revise the patch to fix these failures.\n"
            )

        llm = get_llm(tier=ModelTier.MID, temperature=0.15)
        human = (
            f"Bug ID: {bug_id}\n"
            f"Branch: {branch_name}\n\n"
            f"## Bug Description\n{state.get('bug_description', '')[:10000]}\n\n"
            f"## Code Analysis\n{state.get('code_analysis_summary', '')[:8000]}\n\n"
            f"## Files to Modify\n{context_blob or '_No file context_'}"
            f"{prior_feedback}{test_feedback}"
        )

        fallback = PatchPlan(
            branch_name=branch_name,
            file_edits=[],
            unit_test_files=[],
            commit_message=f"fix: address bug {bug_id}",
            confidence=0.3,
            rationale="LLM unavailable or parse failure — no edits applied",
        )
        plan = invoke_structured(llm, _PATCH_SYSTEM, human, PatchPlan, fallback=fallback)
        branch_name = plan.branch_name or branch_name
        updates["patch_branch_name"] = branch_name
        updates["fix_confidence"] = plan.confidence

        # --- Apply edits ---
        git = GitHelper(
            repo_path=repo_path,
            github_repo_name=state.get("github_repo_name"),
        )
        try:
            git.create_branch(branch_name)
        except GitHelperError as exc:
            writer.append_thought(AGENT_NAME, f"Branch create warning: {exc}")

        applied: list[str] = []
        for edit in plan.file_edits:
            path = edit.get("path")
            content = edit.get("content") or edit.get("content_or_diff") or ""
            action = edit.get("action", "write")
            if not path or not content:
                continue
            target = repo_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if action == "replace" and target.exists():
                # Naive whole-file replace when only a snippet is provided
                target.write_text(content, encoding="utf-8")
            else:
                target.write_text(content, encoding="utf-8")
            applied.append(path)

        for tf in plan.unit_test_files:
            path = tf.get("path")
            content = tf.get("content", "")
            if not path or not content:
                continue
            target = repo_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            applied.append(path)

        writer.append_thought(AGENT_NAME, f"Applied edits to: {applied}")

        # --- Run tests ---
        runner = TestRunner(repo_path)
        run_result = runner.run_pytest()
        writer.append_thought(
            AGENT_NAME,
            f"pytest via {run_result.python_bin} (fallback={run_result.used_fallback}) "
            f"exit={run_result.exit_code}",
        )
        eval_llm = get_llm(tier=ModelTier.LOW, temperature=0.0)
        evaluation = invoke_structured(
            eval_llm,
            _TEST_EVAL_SYSTEM,
            f"Exit code: {run_result.exit_code}\n\nOutput:\n{run_result.output[:12000]}",
            TestEvaluation,
            fallback=TestEvaluation(
                passed=run_result.passed,
                failure_summary="" if run_result.passed else run_result.output[:1000],
                suggested_fixes=[],
                confidence=0.8 if run_result.passed else 0.5,
            ),
        )

        # Prefer actual exit code over LLM opinion
        passed = run_result.passed
        test_results: TestResults = {
            "passed": passed,
            "output": run_result.output,
            "attempts": attempt,
            "exit_code": run_result.exit_code,
        }
        updates["test_results"] = test_results

        if evaluation.suggested_fixes and not passed:
            writer.append_thought(
                AGENT_NAME, f"Test eval suggestions: {evaluation.suggested_fixes}"
            )

        pr_url = state.get("pr_url") or ""
        pr_number = state.get("pr_number")
        diff_text = ""

        confidence_ok = plan.confidence >= settings.confidence_threshold
        if passed and confidence_ok and applied:
            git.stage_all(applied)
            git.commit(plan.commit_message)
            diff_text = git.get_diff()
            updates["patch_diff"] = diff_text
            try:
                git.push(branch_name)
                pr = git.create_pull_request(
                    title=f"fix: {bug_id} — automated patch",
                    body=_pr_body(state, plan, run_result.output),
                    head_branch=branch_name,
                )
                pr_url = pr["html_url"]
                pr_number = pr["number"]
                updates["pr_url"] = pr_url
                updates["pr_number"] = pr_number
                writer.append_thought(AGENT_NAME, f"Opened PR {pr_url}")
            except GitHelperError as exc:
                msg = f"Git push/PR failed: {exc}"
                writer.append_thought(AGENT_NAME, msg)
                updates.setdefault("error_logs", []).append(f"{AGENT_NAME}: {msg}")
                # Keep local commit; PR may be created manually
                diff_text = diff_text or git.get_diff()
                updates["patch_diff"] = diff_text
        elif not applied:
            updates.setdefault("error_logs", []).append(
                f"{AGENT_NAME}: No file edits produced by the model"
            )
            updates["fix_confidence"] = min(plan.confidence, 0.4)
        elif not passed:
            writer.append_thought(AGENT_NAME, f"Tests failed on attempt {attempt}")
            # Store failure output for next retry loop
            updates["fix_confidence"] = min(plan.confidence, 0.55)
        else:
            # Tests passed but confidence below threshold — still commit locally, skip PR
            git.stage_all(applied)
            git.commit(plan.commit_message + " [low-confidence]")
            updates["patch_diff"] = git.get_diff()
            updates.setdefault("error_logs", []).append(
                f"{AGENT_NAME}: fix_confidence {plan.confidence:.2f} below threshold; PR skipped"
            )

        writer.write_markdown(
            filename="03_patch_and_test_execution.md",
            title=f"Patch & Test Execution — {bug_id}",
            sections={
                "Attempt": str(attempt),
                "Branch": branch_name,
                "Rationale": plan.rationale,
                "Files Modified": applied or ["_None_"],
                "Commit Message": plan.commit_message,
                "Test Passed": str(passed),
                "Test Exit Code": str(run_result.exit_code),
                "Test Output": f"```\n{run_result.output[-12000:]}\n```",
                "Test Evaluation": evaluation.failure_summary or "_Passed / N/A_",
                "Diff": f"```diff\n{(updates.get('patch_diff') or diff_text or '(no diff)')[:15000]}\n```",
                "Pull Request": pr_url or "_Not created_",
            },
            metadata={
                "bug_id": bug_id,
                "fix_confidence": updates.get("fix_confidence", plan.confidence),
                "attempt": attempt,
                "max_retries": settings.max_fix_retries,
            },
        )
        return updates

    except QuotaExceededError as exc:
        logger.error("%s quota exceeded: %s", AGENT_NAME, exc)
        writer.write_error_artifact(AGENT_NAME, str(exc), context="QUOTA_EXCEEDED")
        return {
            **updates,
            "test_results": {
                "passed": False,
                "output": str(exc),
                "attempts": attempt,
                "exit_code": -1,
            },
            "fix_confidence": 0.0,
            "error_logs": [f"QUOTA_EXCEEDED: {AGENT_NAME}: {exc}"],
            "pipeline_status": "HALTED",
        }

    except Exception as exc:
        logger.exception("%s failed", AGENT_NAME)
        writer.write_error_artifact(AGENT_NAME, str(exc))
        return {
            **updates,
            "test_results": {
                "passed": False,
                "output": str(exc),
                "attempts": attempt,
                "exit_code": -1,
            },
            "fix_confidence": 0.0,
            "error_logs": [f"{AGENT_NAME}: {exc}"],
        }


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()[:40] or "unknown"


def _pr_body(state: SDLCState, plan: PatchPlan, test_output: str) -> str:
    return (
        f"## Automated Fix for `{state.get('bug_id')}`\n\n"
        f"{plan.rationale}\n\n"
        f"### Analysis Summary\n{state.get('code_analysis_summary', '')[:3000]}\n\n"
        f"### Confidence\n- Fix: {plan.confidence:.2f}\n\n"
        f"### Test Output (truncated)\n```\n{test_output[-3000:]}\n```\n\n"
        f"_Generated by SDLC Orchestrator Agent 3._\n"
    )
