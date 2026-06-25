#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
max_build_num=0

for f in "$ROOT_DIR"/doc-build-*.log; do
    name="$(basename "$f")"
    if [[ "$name" =~ ^doc-build-([0-9]+)\.log$ ]]; then
        n="${BASH_REMATCH[1]}"
        (( n > max_build_num )) && max_build_num="$n"
    fi
done

next_build_num=$((max_build_num + 1))
DOC_BUILD_LOG_FILE="$ROOT_DIR/doc-build-${next_build_num}.log"

# Optional: show which log file will be used
echo "Using log file: $DOC_BUILD_LOG_FILE"

uv run sphinx-build -W -v -b html docs docs/_build/html 2>&1 | tee "$DOC_BUILD_LOG_FILE"

pushd "$ROOT_DIR/docs/_build/html"
uv run python -m http.server
popd