#!/usr/bin/env bash

pushd "docs/_build/html"
uv run python -m http.server 8000
popd