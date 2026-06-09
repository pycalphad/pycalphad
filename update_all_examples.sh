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

find . -type d | sort | while read dir; do
    cd "$dir"
    echo -e "\nCurrent directory: $(pwd)"
    notebooks=$(find . -maxdepth 1 -type f -name "*.ipynb" | sort)
    if [ -n "$notebooks" ]; then
        echo "$notebooks" | while read notebook; do
            echo -e "\nUpdating $notebook\n"
            jupyter nbconvert --to notebook --execute "$notebook" --output "$notebook"
        done
    fi
    cd - > /dev/null
done
popd