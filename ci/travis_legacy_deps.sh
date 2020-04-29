#!/bin/bash
echo "!!! inside travis_legacy_deps.sh"
if [ "$TRAVIS_OS_NAME" == "osx" ]; then
wget -nc http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
else
wget -nc http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi
chmod +x miniconda.sh
./miniconda.sh -b -f
echo ". $HOME/miniconda3/etc/profile.d/conda.sh" >> $HOME/.bashrc
. $HOME/miniconda3/etc/profile.d/conda.sh
conda update --yes conda
conda create --yes -n condaenv python=$TRAVIS_PYTHON_VERSION
conda install --yes -n condaenv pip
conda config --add channels conda-forge
echo "!!! Exiting travis_legacy_deps.sh"
