#!/usr/bin/env bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pushd "$ROOT_DIR/examples"
rm -rf .ipynb_checkpoints

currentPythonWarnings=$PYTHONWARNINGS
echo -e "\nTemporarily setting PYTHONWARNINGS to 'ignore' to suppress warnings during notebook execution.\n"
export PYTHONWARNINGS="ignore"

find . -type d | sort | while read dir; do
    if [[ "$dir" == *".ipynb_checkpoints"* ]]; then
        echo "Skipping $dir (contains .ipynb_checkpoints)"
        continue
    fi
    cd "$dir"
    echo -e "\nCurrent directory: $(pwd)"
    notebooks=$(find . -maxdepth 1 -type f -name "*.ipynb" | sort)
    if [ -n "$notebooks" ]; then
        echo "$notebooks" | while read notebook; do
            $ROOT_DIR/tools/execute-example-notebook.sh "$notebook"
        done
    fi
    cd - > /dev/null
done
popd

echo -e "\nFinished updating all notebooks in the examples directory.\n"
echo -e "Restoring original PYTHONWARNINGS value: '$currentPythonWarnings'\n"
export PYTHONWARNINGS="$currentPythonWarnings"