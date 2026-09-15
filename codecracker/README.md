# CodeCracker

Hands-on architectural understanding of **third-party / legacy** codebases.

Point it at any public GitHub repo. It clones the tree, parses code for imports and
structure, builds a structural map, then synthesizes:

- Architecture overview
- Data-flow narrative
- Mermaid.js diagram
- **Guided file-by-file walkthrough** (what to open next and what to look for)
- A generated `README.md` you can share with teammates
- A **per-run execution efficiency** table for the cracked target

Designed as a teaching demo of how engineers (and agents) should approach unfamiliar code — not by dumping the whole repo into a context window, but by **establishing a guided tour**.

---

## Architecture & Data Flow of the Tool

```
[ GitHub Public Repo URL ]
            │
            ▼
┌──────────────────────────────────────┐
│ 1. Git Cloner & AST Parser Module    │
│    (Filters out non-code, captures   │
│     imports & module call-graphs)    │
└──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 2. Context Builder & Structural Map  │
│    (Generates folder tree & metadata)│
└──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 3. LLM Prompt Strategy Engine        │
│    (Generates Architecture Summary,  │
│     Mermaid.js Flow, & Guided Tour)  │
└──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 4. Output Generator                  │
│    (Writes generated README.md)      │
└──────────────────────────────────────┘
```

| Stage | Module | Responsibility |
|-------|--------|----------------|
| 1 | `cloner.py` + `ast_parser.py` | Shallow-clone with **hooks disabled** (`core.hooksPath` → `/dev/null`), path-safe dest under `.repos/`, optional `GITHUB_TOKEN` + clone cooldown for rate limits; skip vendor/build artifacts; perform native AST parsing for Python and lightweight regex/heuristic static analysis for polyglot languages (JS/TS/Go/Java/Rust/Ruby). |
| 2 | `context_builder.py` | ASCII folder tree, entrypoint detection, import-graph centrality ranking, packed LLM context JSON with a **tiktoken Token Budgeting Engine** that trims payload sections to a max-token threshold before serialize |
| 3 | `llm_engine.py` | Prompt strategy → architecture + Mermaid + guided tour. Uses OpenAI / Anthropic when keys exist; otherwise a deterministic **heuristic** synthesizer (always works offline) |
| 4 | `output_generator.py` | Jinja-rendered `README.md` (incl. runtime benchmark table), plus `report.json` and `prompt_context.txt` |

---

## Quick start

```bash
cd engineering-playground/codecracker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Works offline (heuristic mode) — no API key required
python -m codecracker crack https://github.com/<owner>/<repo>

# Or analyze a local checkout
python -m codecracker crack https://github.com/<owner>/<repo> --local /path/to/checkout
```

Output lands in `output/<owner>__<repo>/`:

```
output/<owner>__<repo>/
├── README.md            # architecture + guided tour + runtime benchmark
├── report.json          # machine-readable tour + benchmark
└── prompt_context.txt   # exact context fed to the LLM strategy
```

Re-print a tour later:

```bash
python -m codecracker tour output/<owner>__<repo>/report.json
```

### Tests & CI

```bash
pytest tests/test_ast_parser.py tests/test_context_builder.py tests/test_cloner.py tests/test_benchmark.py tests/test_token_budget.py
```

GitHub Actions (`.github/workflows/ci.yml`) runs those tests on Python 3.11 and 3.12 for every push/PR touching `codecracker/`.

### Optional LLM backends

```bash
cp .env.example .env
# set OPENAI_API_KEY or ANTHROPIC_API_KEY
# optional: GITHUB_TOKEN for higher clone rate limits

python -m codecracker crack https://github.com/<owner>/<repo> --provider openai
python -m codecracker crack https://github.com/<owner>/<repo> --provider anthropic
```

---

## Guided walkthrough — what you get

Each generated README includes ordered steps like:

1. Open the package manifest / README — orient on purpose & stack  
2. Open the detected entrypoint (`cli.py`, `main.go`, `index.ts`, …)  
3. Follow imports into API / domain / data layers  
4. At every stop: **why this file**, **what to look for**, **what to open next**

That file-by-file sequence is the core teaching point: *architecture understanding is a path, not a dump*.

---

## CLI reference

```
python -m codecracker crack <github-url> [OPTIONS]

Options:
  -p, --provider [auto|openai|anthropic|heuristic]
  -o, --out PATH              Custom output directory
      --local PATH            Skip clone; use existing checkout
      --max-files INT         Cap parsed source files (default 400)
      --max-tour INT          Cap guided tour steps (default 25)
      --full-clone            Disable shallow clone
```

---

## Key files in *this* project

| Path | Role |
|------|------|
| `codecracker/cloner.py` | Normalize GitHub URLs; hook-safe shallow clone / refresh; path containment; clone rate-limit |
| `codecracker/ast_parser.py` | Filter non-code; Python AST + multi-lang heuristics; import edges |
| `codecracker/context_builder.py` | Tree, entrypoints, key-file ranking, tiktoken token-budget packing |
| `codecracker/llm_engine.py` | Prompt strategy + OpenAI/Anthropic/heuristic synthesizers |
| `codecracker/benchmark.py` | Per-run timing + token-cost estimates for the cracked target |
| `codecracker/output_generator.py` | README / JSON / prompt dump (incl. efficiency table) |
| `codecracker/pipeline.py` | Wires stages 1→4 and records stage timings |
| `codecracker/cli.py` | Typer CLI (`crack`, `tour`) |
| `codecracker/models.py` | Shared dataclasses (`RepoMap`, `GuidedStep`, `BenchmarkStats`, …) |

---

## Design notes

1. **Filter first** — big repos drown you; skip vendor/build and cap file count.  
2. **Structure before prose** — tree + import graph beats raw file contents for orientation.  
3. **Prompt with a map, not a monologue** — Stage 3 receives ranked key files + adjacency, not the whole tree. A **tiktoken Token Budgeting Engine** counts tokens and progressively trims graph/tree/key briefs before serialize so the payload stays under `CODECRACKER_CONTEXT_TOKEN_BUDGET`.  
4. **Always have an offline path** — heuristic mode proves the pipeline without API spend.  
5. **Tour > summary** — summaries fade; an ordered reading path is how humans onboard.
6. **AST Parsing Trade-offs & Polyglot Roadmap** — Native Python AST parsing provides 100% exact import-edge graphs. For multi-language support (JS/TS/Go/Java/Rust/Ruby), the demo relies on lightweight static heuristics to keep external C-binding dependencies zero-install. For production/enterprise deployments, `ast_parser.py` is designed to swap in `tree-sitter` bindings for full incremental AST parsing across all major languages without breaking downstream graph-builder interfaces.

### Execution efficiency (runtime)

Each cracked-repo README gets a live table for **that run** (also under `benchmark` in `report.json`):

| Target Repo | Files Analyzed | Time (Heuristic) | Time (LLM) | Approx. Token Cost |
|-------------|----------------|------------------|------------|--------------------|

Wired by `pipeline.py` → `benchmark.py` → `output_generator.py`. Heuristic times are measured wall-clock (map → synthesize → write; clone excluded). When running offline, LLM time/cost are estimated from packed prompt size so the table still fills both columns.
