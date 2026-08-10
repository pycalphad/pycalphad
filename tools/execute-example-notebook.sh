#!/usr/bin/env bash

echo -e "\nUpdating current example...\n"$1"\n"

currentPythonWarnings=$PYTHONWARNINGS
echo -e "\nTemporarily setting PYTHONWARNINGS to 'ignore' to suppress warnings during notebook execution.\n"
export PYTHONWARNINGS="ignore"

cd "$(dirname "$1")"
pwd

uv run jupyter nbconvert --to notebook --execute "$1" --output "$1"
uv run nb-clean clean "$1" --preserve-notebook-metadata --preserve-cell-outputs --preserve-execution-counts --remove-empty-cells

echo -e "\nFinished updating all notebooks in the examples directory.\n"
echo -e "Restoring original PYTHONWARNINGS value: '$currentPythonWarnings'\n"
export PYTHONWARNINGS="$currentPythonWarnings"