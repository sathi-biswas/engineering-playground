#!/usr/bin/env bash
# Demo: crack a small public repo with the offline heuristic engine.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m codecracker crack https://github.com/pallets/flask --provider heuristic --max-files 200
echo
echo "Open: output/pallets__flask/README.md"
