# Automated AI SDLC Orchestrator

Production-ready Python microservice that coordinates **four specialized AI agents** through a stateful **LangGraph** workflow to ingest bug reports (Google Drive RAG), analyze repositories, generate patches with tests, open GitHub PRs, and perform automated code review.

```
Bug Report (GDrive) → Analyze Code → Fix + Test + PR → Code Review
         Agent 1            Agent 2         Agent 3         Agent 4
```

## Architecture

| Component | Role | Model Tier (FinOps) |
|-----------|------|---------------------|
| **Agent 1** `agent1_bug_reader` | Ingest bug specs via Google Drive RAG | Low / Mid |
| **Agent 2** `agent2_code_analyzer` | AST / file-tree analysis & root-cause | Mid / High |
| **Agent 3** `agent3_bug_fixer` | Patch, unit tests, pytest, GitHub PR | Mid (code) / Low (test eval) |
| **Agent 4** `agent4_code_reviewer` | PR review, inline comments, approve/reject | High |

**Confidence gate:** analysis / fix / review must reach **≥ 0.70** (configurable). Failed tests retry Agent 3 up to **2** times. `NEEDS_REVISION` can loop back to Agent 3.

Artifacts are written as structured Markdown under [`output/`](output/).

## Project Layout

```
├── config.py                 # Settings, API keys, model tiers, paths
├── state.py                  # SDLCState TypedDict & Pydantic schemas
├── graph.py                  # LangGraph orchestration & routers
├── main.py                   # CLI + optional FastAPI trigger
├── requirements.txt
├── utils/
│   ├── logger.py             # Thought log + /output artifact writer
│   ├── git_helper.py         # Branch, commit, push, PR, review
│   ├── test_runner.py        # Isolated pytest subprocess runner
│   └── model_router.py       # FinOps LLM factory (Gemini / OpenAI / Anthropic)
├── tools/
│   ├── gdrive_rag.py         # Drive auth, loader, FAISS/Chroma RAG
│   └── code_parser.py        # AST + file tree analyzer
└── agents/
    ├── agent1_bug_reader.py
    ├── agent2_code_analyzer.py
    ├── agent3_bug_fixer.py
    └── agent4_code_reviewer.py
```

## Prerequisites

- Python **3.11+**
- A local **git** checkout of the target repository
- API keys (see below)
- Optional: Google Cloud service account with Drive read access
- Optional: `GITHUB_TOKEN` with `repo` scope for PR create/review

## Setup

```bash
cd sdlcorchestrator
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then edit values
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes* | Free key from [Google AI Studio](https://aistudio.google.com/apikey) |
| `LLM_PROVIDER` | No | `gemini` (default), `openai`, or `anthropic` |
| `OPENAI_API_KEY` | No | Only if `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | No | Only if `LLM_PROVIDER=anthropic` |
| `GITHUB_TOKEN` | For PRs | Fine-grained or classic token with PR permissions |
| `GOOGLE_APPLICATION_CREDENTIALS` | For Drive | Path to OAuth Desktop or service-account JSON |
| `GDRIVE_FOLDER_ID` | Optional | Default Drive folder for bulk load |
| `GDRIVE_BUG_FILE_ID` | Optional | Default bug-report Drive file ID |
| `GITHUB_REPO_NAME` | Optional | Default `owner/repo` |
| `TARGET_REPO_PATH` | Optional | Default local repo path |
| `MODEL_LOW` / `MODEL_MID` / `MODEL_HIGH` | No | Override FinOps model IDs |
| `PYTEST_ARGS` | No | Extra pytest flags (default `-q --tb=short`) |
| `TARGET_PYTHON_BIN` | No | Python in the *target* repo venv; auto-detects `<repo>/.venv/bin/python` if unset |

\*Without a usable LLM key the pipeline runs in **StubLLM dry-run** mode.

### Model Tier Defaults (Gemini free tier)

Prefer **flash-lite** for local iteration. Full **flash** models are often capped at ~**20 requests/day** on the free tier (per project, resets midnight Pacific), which a single multi-agent run can exhaust. Flash-lite is typically ~**500 RPD**.

| Tier | Gemini default | Used for |
|------|----------------|----------|
| Low | `gemini-3.5-flash-lite` | Parsing, test-output eval |
| Mid | `gemini-3.5-flash-lite` | RAG correlation, patches |
| High | `gemini-3.5-flash-lite` | Final review / edge cases |

Quota tips:
- Limits are **per Google Cloud / AI Studio project**, not per API key.
- On `429 ResourceExhausted`, the orchestrator fails fast (no second invoke, max 1 client retry).
- Override via `MODEL_LOW` / `MODEL_MID` / `MODEL_HIGH` if AI Studio lists a different lite model ID.
- See [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

### Google Drive

1. Create a GCP service account and download JSON key.
2. Share the bug-report Drive file/folder with the service-account email.
3. Set `GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/sa.json`.

**Offline / demo:** pass a local Markdown file path as `--gdrive-file-id` — no Drive credentials needed.

## Usage

### CLI (one-shot)

```bash
python main.py \
  --bug-id BUG-1042 \
  --gdrive-file-id ./samples/sample_bug_report.md \
  --repo-path /path/to/target/repo \
  --github-repo your-org/your-repo \
  --dump-state
```

### FastAPI trigger

```bash
python main.py --bug-id unused --gdrive-file-id unused --serve --port 8080
```

```bash
curl -X POST http://localhost:8080/pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{
    "bug_id": "BUG-1042",
    "gdrive_file_id": "./samples/sample_bug_report.md",
    "target_repo_path": "/path/to/target/repo",
    "github_repo_name": "your-org/your-repo"
  }'
```

Health: `GET /health` · Graph nodes: `GET /graph`

## Output Artifacts

| File | Producer |
|------|----------|
| `output/01_bug_description.md` | Agent 1 |
| `output/02_code_analysis.md` | Agent 2 |
| `output/03_patch_and_test_execution.md` | Agent 3 |
| `output/04_code_review_decision.md` | Agent 4 |
| `output/05_pipeline_summary.md` | Finalize |
| `output/00_pipeline_halted.md` | Error handler |
| `output/thought_process.log` | All agents |

## Routing & Error Handling

1. After Agent 2: if `analysis_confidence < 0.70` → `error_handler`.
2. After Agent 3: if tests fail and `fix_attempt < max_fix_retries` → retry Agent 3; else → `error_handler`.
3. After Agent 4: if `NEEDS_REVISION` and revision loop enabled → Agent 3; if `REJECTED` → `error_handler`; if `APPROVED` → `finalize`.
4. Every node catches exceptions, appends to `error_logs`, and writes fallback error artifacts under `/output`.

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Completed (approved or finalized without hard failure) |
| `1` | Pipeline halted / failed |
| `2` | Review rejected |

## License

Internal engineering playground — adapt as needed for your org.
