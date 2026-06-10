#!/bin/bash
# make sure to run this script from the root directory of the project

if [ -f .venv/bin/activate ]; then
    # for Unix-like OS, e.g. Linux, macOS
    source .venv/bin/activate
else
    # for Windows
    source .venv/Scripts/activate
fi

pushd examples
rm -rf .ipynb_checkpoints

currentPythonWarnings=$PYTHONWARNINGS
echo -e "\nTemporarily setting PYTHONWARNINGS to 'ignore' to suppress warnings during notebook execution.\n"
export PYTHONWARNINGS="ignore"

find . -type d | sort | while read dir; do
    cd "$dir"
    echo -e "\nCurrent directory: $(pwd)"
    notebooks=$(find . -maxdepth 1 -type f -name "*.ipynb" | sort)
    if [ -n "$notebooks" ]; then
        echo "$notebooks" | while read notebook; do
            echo -e "\nUpdating $notebook\n"
            jupyter nbconvert --to notebook --execute "$notebook" --output "$notebook"
            nb-clean clean "$notebook" --remove-all-notebook-metadata --preserve-cell-outputs --preserve-execution-counts --remove-empty-cells
        done
    fi
    cd - > /dev/null
done
popd

echo -e "\nFinished updating all notebooks in the examples directory.\n"
echo -e "Restoring original PYTHONWARNINGS value: '$currentPythonWarnings'\n"
export PYTHONWARNINGS="$currentPythonWarnings"