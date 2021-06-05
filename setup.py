import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import pathlib
import toml

config = toml.load(pathlib.Path(__file__).parent / 'pyproject.toml')

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

CYTHON_COMPILER_DIRECTIVES = {
    "language_level": 3,
}

CYTHON_EXTENSION_INCLUDES = ['.', np.get_include()]
CYTHON_EXTENSION_MODULES = [
    Extension('pycalphad.core.hyperplane', sources=['pycalphad/core/hyperplane.pyx']),
    Extension('pycalphad.core.eqsolver', sources=['pycalphad/core/eqsolver.pyx']),
    Extension('pycalphad.core.phase_rec', sources=['pycalphad/core/phase_rec.pyx']),
    Extension('pycalphad.core.composition_set', sources=['pycalphad/core/composition_set.pyx']),
    Extension('pycalphad.core.problem', sources=['pycalphad/core/problem.pyx']),
    Extension('pycalphad.core.minimizer', sources=['pycalphad/core/minimizer.pyx']),
]

setup(
    name=config['project']['name'],
    author=config['project']['authors'][0]['name'],
    author_email=config['project']['authors'][0]['email'],
    description=config['project']['description'],
    # Do NOT include pycalphad._dev here. It is for local development and should not be distributed.
    packages=['pycalphad', 'pycalphad.codegen', 'pycalphad.core', 'pycalphad.io', 'pycalphad.plot', 'pycalphad.plot.binary', 'pycalphad.tests'],
    ext_modules=cythonize(
        CYTHON_EXTENSION_MODULES,
        include_path=CYTHON_EXTENSION_INCLUDES,
        compiler_directives=CYTHON_COMPILER_DIRECTIVES,
    ),
    package_data={
        'pycalphad/core': ['*.pxd'],
    },
    # This include is for the compiler to find the *.h files during the build_ext phase
    # the include must contain a symengine directory with header files
    include_dirs=[np.get_include()],
    license=config['project']['license'],
    long_description=read(config['project']['readme']),
    long_description_content_type='text/x-rst',
    url=config['project']['urls']['homepage'],
    install_requires=config['project']['dependencies'],
    classifiers=config['project']['classifiers'],
)
