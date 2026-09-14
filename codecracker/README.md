# CodeCracker

Hands-on architectural understanding of **third-party / legacy** codebases.

Point it at any public GitHub repo. It clones the tree, parses code for imports and
structure, builds a structural map, then synthesizes:

- Architecture overview
- Data-flow narrative
- Mermaid.js diagram
- **Guided file-by-file walkthrough** (what to open next and what to look for)
- A generated `README.md` you can share with teammates

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
| 1 | `cloner.py` + `ast_parser.py` | Shallow-clone the repo; skip `node_modules` / build artifacts; AST-parse Python and heuristically parse JS/TS/Go/Java/Rust/Ruby for imports, classes, functions |
| 2 | `context_builder.py` | ASCII folder tree, entrypoint detection, import-graph centrality ranking, packed LLM context JSON |
| 3 | `llm_engine.py` | Prompt strategy → architecture + Mermaid + guided tour. Uses OpenAI / Anthropic when keys exist; otherwise a deterministic **heuristic** synthesizer (always works offline) |
| 4 | `output_generator.py` | Jinja-rendered `README.md`, plus `report.json` and `prompt_context.txt` |

---

## Quick start

```bash
cd engineering-playground/codecracker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Works offline (heuristic mode) — no API key required
python -m codecracker crack https://github.com/pallets/flask

# Or analyze a local checkout
python -m codecracker crack https://github.com/owner/repo --local /path/to/checkout
```

### Tests & CI

Focused unit tests cover the parser and structural map stages:

```bash
pytest tests/test_ast_parser.py tests/test_context_builder.py
```

GitHub Actions (`.github/workflows/ci.yml`) runs those tests on Python 3.11 and 3.12 for every push/PR touching `codecracker/`.

Output lands in `output/<owner>__<repo>/`:

```
output/pallets__flask/
├── README.md            # architecture + guided tour
├── report.json          # machine-readable tour
└── prompt_context.txt   # exact context fed to the LLM strategy
```

Re-print a tour later:

```bash
python -m codecracker tour output/pallets__flask/report.json
```

### Optional LLM backends

```bash
cp .env.example .env
# set OPENAI_API_KEY or ANTHROPIC_API_KEY

python -m codecracker crack https://github.com/encode/httpx --provider openai
python -m codecracker crack https://github.com/encode/httpx --provider anthropic
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
| `codecracker/cloner.py` | Normalize GitHub URLs; shallow clone / refresh |
| `codecracker/ast_parser.py` | Filter non-code; Python AST + multi-lang heuristics; import edges |
| `codecracker/context_builder.py` | Tree, entrypoints, key-file ranking, LLM context pack |
| `codecracker/llm_engine.py` | Prompt strategy + OpenAI/Anthropic/heuristic synthesizers |
| `codecracker/output_generator.py` | README / JSON / prompt dump |
| `codecracker/pipeline.py` | Wires stages 1→4 |
| `codecracker/cli.py` | Typer CLI (`crack`, `tour`) |
| `codecracker/models.py` | Shared dataclasses (`RepoMap`, `GuidedStep`, …) |

---

## Design notes (for the demo narrative)

1. **Filter first** — big repos drown you; skip vendor/build and cap file count.  
2. **Structure before prose** — tree + import graph beats raw file contents for orientation.  
3. **Prompt with a map, not a monologue** — Stage 3 receives ranked key files + adjacency, not the whole tree.  
4. **Always have an offline path** — heuristic mode proves the pipeline without API spend.  
5. **Tour > summary** — summaries fade; an ordered reading path is how humans onboard.

---

## Example targets

Good demos (public, approachable size):

- `https://github.com/pallets/flask`
- `https://github.com/encode/httpx`
- `https://github.com/psf/requests`
- `https://github.com/tiangolo/fastapi`
