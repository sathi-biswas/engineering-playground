"""Runtime execution-efficiency benchmarks for a cracked repo."""

from __future__ import annotations

from .models import ArchitectureReport, BenchmarkStats, RepoMap

# Ballpark gpt-4o-mini pricing used when the API does not return usage.
_USD_PER_M_INPUT = 0.15
_USD_PER_M_OUTPUT = 0.60
_EST_OUTPUT_TOKENS = 2000
_EST_API_LATENCY_S = 2.5
_EST_GEN_TOKENS_PER_S = 800


def estimate_tokens(text: str) -> int:
    """Rough char/4 token estimate (good enough for cost display)."""
    return max(1, len(text) // 4)


def estimate_llm_cost_usd(input_tokens: int, output_tokens: int = _EST_OUTPUT_TOKENS) -> float:
    return (input_tokens / 1_000_000) * _USD_PER_M_INPUT + (
        output_tokens / 1_000_000
    ) * _USD_PER_M_OUTPUT


def estimate_llm_wall_s(map_s: float, output_tokens: int = _EST_OUTPUT_TOKENS) -> float:
    """Map time + typical API latency + generation."""
    return map_s + _EST_API_LATENCY_S + (output_tokens / _EST_GEN_TOKENS_PER_S)


def build_benchmark(
    repo_map: RepoMap,
    report: ArchitectureReport,
    *,
    clone_s: float,
    map_s: float,
    synth_s: float,
    write_s: float,
) -> BenchmarkStats:
    """
    Build the per-run benchmark row for the cracked target.

    - Heuristic provider: ``Time (Heuristic)`` is measured (map+synth+write);
      ``Time (LLM)`` / cost are estimates from the packed prompt.
    - LLM provider: ``Time (LLM)`` is measured synth (+ map/write in total
      display uses synth for the LLM column); heuristic column estimates
      map+write with negligible local synth.
    """
    prompt_chars = len(report.raw_prompt or "")
    est_in = estimate_tokens(report.raw_prompt or "")
    est_out = _EST_OUTPUT_TOKENS
    cost = estimate_llm_cost_usd(est_in, est_out)

    analyze_s = map_s + synth_s + write_s  # clone excluded (network-variable)

    if report.provider == "heuristic":
        time_heuristic = round(analyze_s, 3)
        time_llm = round(estimate_llm_wall_s(map_s, est_out), 2)
        notes = (
            "Heuristic time is measured wall-clock (map → synthesize → write; "
            "clone excluded). LLM time/cost estimated for gpt-4o-mini from "
            "packed prompt size (~4 chars/token, ~2K output tokens)."
        )
    else:
        time_llm = round(map_s + synth_s + write_s, 3)
        # Local heuristic synth is effectively free vs map/write
        time_heuristic = round(map_s + write_s + 0.01, 3)
        notes = (
            f"LLM time is measured for provider `{report.provider}` "
            "(map → synthesize → write; clone excluded). Heuristic time "
            "estimated as map+write. Token cost estimated from prompt size "
            "(API usage not required)."
        )

    return BenchmarkStats(
        target_repo=repo_map.repo_name,
        files_analyzed=len(repo_map.files),
        time_heuristic_s=time_heuristic,
        time_llm_s=time_llm,
        approx_token_cost_usd=round(cost, 4),
        provider=report.provider,
        clone_s=round(clone_s, 3),
        map_s=round(map_s, 3),
        synth_s=round(synth_s, 3),
        write_s=round(write_s, 3),
        prompt_chars=prompt_chars,
        est_input_tokens=est_in,
        est_output_tokens=est_out,
        notes=notes,
    )


def format_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.01:
        return "<0.01s"
    if value < 10:
        return f"{value:.2f}s"
    return f"{value:.1f}s"


def format_cost(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.3f}"
