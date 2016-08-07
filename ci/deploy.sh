#!/bin/bash
# Based on https://gist.github.com/domenic/ec8b0fc8ab45f39403dd
set -e # Exit with nonzero exit code if anything fails

# Deploy to 'latest' on every commit to develop
SOURCE_LATEST_BRANCH="develop"
# Deploy only on tagged commits to master
SOURCE_TAG_BRANCH="master"
TARGET_BRANCH="gh-pages"

function doCompile {
  # Need to think more about API doc rebuild because that lives outside gh-pages
  # sphinx-apidoc -o docs/api/ .
  sphinx-build -b html docs docs/_build/html
}

if [ "$SOURCE_LATEST_BRANCH" = "$TRAVIS_BRANCH" ] && [ "$TRAVIS_TAG" = "" ]; then
   DEPLOY_NAME="latest"
fi

if [ "$SOURCE_TAG_BRANCH" = "$TRAVIS_BRANCH" ] && [ "$TRAVIS_TAG" != "" ]; then
   DEPLOY_NAME="$TRAVIS_TAG"
fi

# Pull requests and commits to other branches shouldn't try to deploy, just build to verify
if [ "$TRAVIS_PULL_REQUEST" != "false" ] || [ "DEPLOY_NAME" != "" ] || [ "$DEPLOY_ENC_LABEL" == "" ]; then
    echo "Skipping deploy; just doing a docs build."
    doCompile
    exit 0
fi

# Save some useful information
REPO=`git config remote.origin.url`
SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:}
SHA=`git rev-parse --verify HEAD`

# Clone the existing gh-pages for this repo into out/
# Create a new empty branch if gh-pages doesn't exist yet (should only happen on first deploy)
git clone $REPO out
cd out
git checkout $TARGET_BRANCH || git checkout --orphan $TARGET_BRANCH
cd ..

# Clean out any existing contents
mkdir -p out/docs/$DEPLOY_NAME
rm -rf out/docs/$DEPLOY_NAME/* || exit 0

# Run our compile script
doCompile
cp -Rf docs/_build/html out/docs/$DEPLOY_NAME

# Now let's go have some fun with the cloned repo
cd out
git config user.name "Travis CI"
git config user.email "$COMMIT_AUTHOR_EMAIL"

# If there are no changes to the compiled out (e.g. this is a README update) then just bail.
if [ $(git status --porcelain | wc -l) -lt 1 ]; then
    echo "No changes to the output on this push; exiting."
    exit 0
fi

# Commit the "changes", i.e. the new version.
# The delta will show diffs between new and old versions.
git add -A .
git commit -m "DOC: Deploy '${DEPLOY_NAME}' docs to GitHub Pages: ${SHA}"

# Get the deploy key by using Travis's stored variables to decrypt deploy_key.enc
ENCRYPTED_KEY_VAR="encrypted_${DEPLOY_ENC_LABEL}_key"
ENCRYPTED_IV_VAR="encrypted_${DEPLOY_ENC_LABEL}_iv"
ENCRYPTED_KEY=${!ENCRYPTED_KEY_VAR}
ENCRYPTED_IV=${!ENCRYPTED_IV_VAR}
openssl aes-256-cbc -K $ENCRYPTED_KEY -iv $ENCRYPTED_IV -in ../ci/deploy_key.enc -out ../ci/deploy_key -d
chmod 600 ../ci/deploy_key
eval `ssh-agent -s`
ssh-add ../ci/deploy_key

# Now that we're all set up, we can push.
git push $SSH_REPO $TARGET_BRANCH