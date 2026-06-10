#!/bin/bash

if [ -f .venv/bin/activate ]; then
    # for Unix-like OS, e.g. Linux, macOS
    source .venv/bin/activate
else
    # for Windows
    source .venv/Scripts/activate
fi

echo -e "\nUpdating current example...\n"$1"\n"

currentPythonWarnings=$PYTHONWARNINGS
echo -e "\nTemporarily setting PYTHONWARNINGS to 'ignore' to suppress warnings during notebook execution.\n"
export PYTHONWARNINGS="ignore"

cd "$(dirname "$1")"
pwd

jupyter nbconvert --to notebook --execute "$1" --output "$1"
nb-clean clean "$1" --preserve-notebook-metadata --preserve-cell-outputs --preserve-execution-counts --remove-empty-cells

echo -e "\nFinished updating all notebooks in the examples directory.\n"
echo -e "Restoring original PYTHONWARNINGS value: '$currentPythonWarnings'\n"
export PYTHONWARNINGS="$currentPythonWarnings"