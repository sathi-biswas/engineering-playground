#!/usr/bin/env python3
"""CLI (and optional FastAPI) entrypoint for the SDLC Orchestrator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from config import get_settings
from graph import build_graph, run_pipeline
from state import SDLCState
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Automated AI SDLC Orchestrator — LangGraph multi-agent pipeline",
    )
    parser.add_argument("--bug-id", required=True, help="Unique bug identifier")
    parser.add_argument(
        "--gdrive-file-id",
        required=True,
        help="Google Drive file ID OR local path to a bug-report Markdown/text file",
    )
    parser.add_argument(
        "--repo-path",
        default=str(settings.default_target_repo_path or "."),
        help="Local path to the target git repository",
    )
    parser.add_argument(
        "--github-repo",
        default=settings.default_github_repo_name or "",
        help="GitHub repo full name (owner/name) for PR creation",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start optional FastAPI trigger server instead of running once",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="FastAPI bind host (with --serve)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="FastAPI bind port (with --serve)",
    )
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Print final pipeline state as JSON to stdout",
    )
    return parser.parse_args(argv)


def build_initial_state(args: argparse.Namespace) -> SDLCState:
    """Map CLI args into an SDLCState seed dict."""
    repo = Path(args.repo_path).expanduser().resolve()
    return {
        "bug_id": args.bug_id,
        "gdrive_file_id": args.gdrive_file_id,
        "target_repo_path": str(repo),
        "github_repo_name": args.github_repo,
        "error_logs": [],
        "affected_files": [],
        "fix_attempt": 0,
        "revision_count": 0,
        "pipeline_status": "RUNNING",
    }


def create_fastapi_app() -> Any:
    """Build a minimal FastAPI app that triggers the pipeline via POST."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="SDLC Orchestrator",
        description="Trigger the LangGraph multi-agent SDLC pipeline",
        version="1.0.0",
    )

    class PipelineRequest(BaseModel):
        bug_id: str
        gdrive_file_id: str
        target_repo_path: str
        github_repo_name: str = ""

    class PipelineResponse(BaseModel):
        pipeline_status: str
        review_decision: str | None = None
        pr_url: str | None = None
        error_logs: list[str] = Field(default_factory=list)
        analysis_confidence: float | None = None
        fix_confidence: float | None = None
        review_confidence: float | None = None

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/pipeline/run", response_model=PipelineResponse)
    def run(req: PipelineRequest) -> PipelineResponse:
        try:
            final = run_pipeline(
                {
                    "bug_id": req.bug_id,
                    "gdrive_file_id": req.gdrive_file_id,
                    "target_repo_path": req.target_repo_path,
                    "github_repo_name": req.github_repo_name,
                    "error_logs": [],
                    "affected_files": [],
                    "fix_attempt": 0,
                    "revision_count": 0,
                    "pipeline_status": "RUNNING",
                }
            )
        except Exception as exc:
            logger.exception("Pipeline HTTP trigger failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return PipelineResponse(
            pipeline_status=str(final.get("pipeline_status", "UNKNOWN")),
            review_decision=final.get("review_decision"),
            pr_url=final.get("pr_url"),
            error_logs=list(final.get("error_logs") or []),
            analysis_confidence=final.get("analysis_confidence"),
            fix_confidence=final.get("fix_confidence"),
            review_confidence=final.get("review_confidence"),
        )

    @app.get("/graph")
    def graph_info() -> dict[str, Any]:
        build_graph()  # validate compile
        return {
            "nodes": [
                "agent1_bug_reader",
                "agent2_code_analyzer",
                "agent3_bug_fixer",
                "agent4_code_reviewer",
                "error_handler",
                "finalize",
            ],
            "entry": "agent1_bug_reader",
        }

    return app


def main(argv: list[str] | None = None) -> int:
    """CLI main."""
    setup_logging()
    args = parse_args(argv)
    settings = get_settings()
    settings.ensure_directories()

    if args.serve:
        import uvicorn

        logger.info("Starting FastAPI server on %s:%s", args.host, args.port)
        uvicorn.run(create_fastapi_app(), host=args.host, port=args.port, log_level="info")
        return 0

    initial = build_initial_state(args)
    logger.info(
        "Running pipeline bug_id=%s repo=%s gdrive=%s",
        initial["bug_id"],
        initial["target_repo_path"],
        initial["gdrive_file_id"],
    )

    final = run_pipeline(initial)

    status = final.get("pipeline_status", "UNKNOWN")
    decision = final.get("review_decision")
    pr_url = final.get("pr_url")
    print("\n========== SDLC Orchestrator Result ==========")
    print(f"Status:     {status}")
    print(f"Decision:   {decision}")
    print(f"PR URL:     {pr_url}")
    print(f"Artifacts:  {settings.output_dir}")
    if final.get("error_logs"):
        print("Errors:")
        for err in final["error_logs"]:
            print(f"  - {err}")
    print("==============================================\n")

    if args.dump_state:
        # Convert non-JSON-native values safely
        print(json.dumps(dict(final), indent=2, default=str))

    if status in {"HALTED", "FAILED"}:
        return 1
    if decision == "REJECTED":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
