#!/bin/bash
# make sure to run this script from the root directory of the project

pushd examples
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
            echo -e "\nUpdating $notebook\n"
            uv run jupyter nbconvert --to notebook --execute "$notebook" --output "$notebook"
            uv run nb-clean clean "$notebook" --preserve-notebook-metadata --preserve-cell-outputs --preserve-execution-counts --remove-empty-cells
        done
    fi
    cd - > /dev/null
done
popd

echo -e "\nFinished updating all notebooks in the examples directory.\n"
echo -e "Restoring original PYTHONWARNINGS value: '$currentPythonWarnings'\n"
export PYTHONWARNINGS="$currentPythonWarnings"