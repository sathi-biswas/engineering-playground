"""Tests for runtime benchmark table included in cracked-repo README."""

from __future__ import annotations

from pathlib import Path

from codecracker.benchmark import build_benchmark, format_cost, format_seconds
from codecracker.models import ArchitectureReport, GuidedStep, RepoMap
from codecracker.output_generator import render_readme, write_outputs
from codecracker.config import Settings


def _sample(repo_name: str = "wallstead/SQLit") -> tuple[RepoMap, ArchitectureReport]:
    root = Path("/tmp/sample")
    repo_map = RepoMap(
        repo_url=f"https://github.com/{repo_name}",
        repo_name=repo_name,
        root=root,
        tree_text="sample/\n└── main.swift\n",
        files=[],
        metadata={"file_count": 4},
    )
    # Pretend 4 files for table display via empty files list — set via Fake
    report = ArchitectureReport(
        overview="A tiny Swift DBMS.",
        architecture_summary="### Layers\n- core",
        data_flow="entrypoint → runner",
        mermaid="flowchart TD\n  A-->B",
        key_files=[{"path": "Sources/pa4/main.swift", "role": "entry"}],
        guided_tour=[
            GuidedStep(
                order=1,
                path="Sources/pa4/main.swift",
                title="Start here",
                why="Entrypoint",
                what_to_look_for=["main"],
                next_hint="Open Command.swift",
            )
        ],
        how_to_extend="Trace SQL parse path.",
        raw_prompt="x" * 400,
        provider="heuristic",
    )
    return repo_map, report


class TestBenchmarkTableInReadme:
    def test_render_includes_efficiency_table(self) -> None:
        repo_map, report = _sample()
        # Simulate 4 analyzed files
        from codecracker.models import FileFacts

        repo_map.files = [
            FileFacts(path=f"f{i}.swift", language="swift", lines=10) for i in range(4)
        ]
        bench = build_benchmark(
            repo_map, report, clone_s=0.5, map_s=0.02, synth_s=0.01, write_s=0.005
        )
        md = render_readme(repo_map, report, bench)
        assert "## Execution efficiency (this run)" in md
        assert "| Target Repo | Files Analyzed | Time (Heuristic) | Time (LLM) | Approx. Token Cost |" in md
        assert "wallstead/SQLit" in md
        assert "| 4 |" in md
        assert format_seconds(bench.time_heuristic_s) in md
        assert format_seconds(bench.time_llm_s) in md
        assert format_cost(bench.approx_token_cost_usd) in md

    def test_write_outputs_persists_benchmark_json(self, tmp_path: Path) -> None:
        import json

        repo_map, report = _sample()
        from codecracker.models import FileFacts

        repo_map.files = [
            FileFacts(path="main.swift", language="swift", lines=10)
        ]
        settings = Settings(work_dir=tmp_path / "repos", output_dir=tmp_path / "out")
        bench = build_benchmark(
            repo_map, report, clone_s=0.1, map_s=0.05, synth_s=0.02, write_s=0.01
        )
        dest = write_outputs(
            repo_map, report, settings, out_dir=tmp_path / "out" / "wallstead__SQLit", benchmark=bench
        )
        readme = (dest / "README.md").read_text(encoding="utf-8")
        assert "Execution efficiency (this run)" in readme
        data = json.loads((dest / "report.json").read_text(encoding="utf-8"))
        assert data["benchmark"]["target_repo"] == "wallstead/SQLit"
        assert data["benchmark"]["files_analyzed"] == 1
        assert "time_heuristic_s" in data["benchmark"]
