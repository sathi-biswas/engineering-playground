"""Agent 2 — Senior Code Architect / Analyzer."""

from __future__ import annotations

from typing import Any

from config import ModelTier, get_settings
from state import CodeAnalysisResult, SDLCState
from tools.code_parser import CodeParser
from utils.logger import get_artifact_writer, get_logger
from utils.model_router import get_llm
from agents.llm_helpers import invoke_structured

logger = get_logger(__name__)

AGENT_NAME = "Agent2_CodeAnalyzer"

_SYSTEM = """You are a Senior Software Architect performing root-cause analysis.
Given a bug description and a repository structure (AST + file tree), identify:
1. Affected source files (relative paths)
2. Specific functions/classes implicated
3. A clear root-cause hypothesis
4. A concrete suggested fix approach
Confidence must be honest (0-1). Only list files that plausibly exist in the tree.
"""


def run_code_analyzer(state: SDLCState) -> dict[str, Any]:
    """LangGraph node: correlate bug report to concrete source locations."""
    writer = get_artifact_writer()
    settings = get_settings()
    writer.append_thought(AGENT_NAME, "Starting codebase analysis")

    updates: dict[str, Any] = {"current_agent": AGENT_NAME}

    try:
        repo_path = state.get("target_repo_path") or ""
        if not repo_path:
            raise ValueError("target_repo_path is required")

        parser = CodeParser(repo_path)
        repo_analysis = parser.analyze_repo()
        tree_context = parser.to_llm_context(repo_analysis)

        # Heuristic candidate files from bug text keywords
        bug_text = state.get("bug_description") or ""
        candidates = _heuristic_candidates(bug_text, repo_analysis.file_tree)
        file_context = parser.gather_context(candidates, max_files=10)
        file_context_blob = "\n\n".join(
            f"### `{path}`\n```\n{src[:4000]}\n```" for path, src in file_context.items()
        )

        writer.append_thought(
            AGENT_NAME,
            f"Repo has {repo_analysis.file_count} files; "
            f"gathered context for {len(file_context)} candidates",
        )

        # Mid/High tier for deep analysis
        llm = get_llm(tier=ModelTier.MID, temperature=0.2)

        human = (
            f"Bug ID: {state.get('bug_id')}\n\n"
            f"## Bug Description\n{bug_text[:12000]}\n\n"
            f"## Repository Structure\n{tree_context}\n\n"
            f"## Candidate File Contents\n{file_context_blob or '_No candidates_'}"
        )

        fallback = CodeAnalysisResult(
            architecture_overview=repo_analysis.summary[:2000],
            root_cause_hypothesis="Unable to determine root cause automatically",
            affected_files=candidates[:5],
            target_functions=[],
            suggested_approach="Manual investigation required",
            confidence=0.35,
        )
        result = invoke_structured(
            llm, _SYSTEM, human, CodeAnalysisResult, fallback=fallback
        )

        # Validate affected files against tree when possible
        known = set(repo_analysis.file_tree)
        validated = [f for f in result.affected_files if f in known or f.lstrip("./") in known]
        if not validated and candidates:
            validated = candidates[:5]
            result.confidence = min(result.confidence, 0.65)

        updates["affected_files"] = validated or result.affected_files
        updates["code_analysis_summary"] = _format_analysis(result, repo_analysis.summary)
        updates["analysis_confidence"] = result.confidence

        writer.write_markdown(
            filename="02_code_analysis.md",
            title=f"Code Analysis — {state.get('bug_id', 'N/A')}",
            sections={
                "Architecture Overview": result.architecture_overview,
                "Root-Cause Hypothesis": result.root_cause_hypothesis,
                "Affected Files": updates["affected_files"] or ["_None identified_"],
                "Target Functions / Classes": result.target_functions or ["_None_"],
                "Suggested Approach": result.suggested_approach,
                "Repository Snapshot": repo_analysis.summary[:8000],
            },
            metadata={
                "bug_id": state.get("bug_id"),
                "analysis_confidence": result.confidence,
                "threshold": settings.confidence_threshold,
                "file_count": repo_analysis.file_count,
                "total_loc": repo_analysis.total_loc,
            },
        )

        writer.append_thought(
            AGENT_NAME,
            f"Analysis complete confidence={result.confidence:.2f} "
            f"files={updates['affected_files']}",
        )
        return updates

    except Exception as exc:
        logger.exception("%s failed", AGENT_NAME)
        writer.write_error_artifact(AGENT_NAME, str(exc))
        return {
            **updates,
            "affected_files": [],
            "code_analysis_summary": f"ERROR: {exc}",
            "analysis_confidence": 0.0,
            "error_logs": [f"{AGENT_NAME}: {exc}"],
        }


def _heuristic_candidates(bug_text: str, file_tree: list[str]) -> list[str]:
    """Score file paths by keyword overlap with the bug description."""
    tokens = {t.lower() for t in bug_text.replace("/", " ").replace(".", " ").split() if len(t) > 3}
    scored: list[tuple[int, str]] = []
    for path in file_tree:
        score = 0
        lower = path.lower()
        basename = lower.rsplit("/", 1)[-1]
        stem = basename.rsplit(".", 1)[0]
        for tok in tokens:
            if tok in lower:
                score += 2
            if tok == stem or tok in stem:
                score += 3
        # Prefer source over tests slightly for analysis focus
        if "test" in lower:
            score = max(0, score - 1)
        if score:
            scored.append((score, path))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [p for _, p in scored[:15]]


def _format_analysis(result: CodeAnalysisResult, repo_summary: str) -> str:
    return (
        f"### Architecture\n{result.architecture_overview}\n\n"
        f"### Root Cause\n{result.root_cause_hypothesis}\n\n"
        f"### Affected Files\n"
        + "\n".join(f"- `{f}`" for f in result.affected_files)
        + f"\n\n### Target Functions\n"
        + "\n".join(f"- `{f}`" for f in result.target_functions)
        + f"\n\n### Suggested Approach\n{result.suggested_approach}\n\n"
        f"### Confidence\n{result.confidence}\n"
    )
