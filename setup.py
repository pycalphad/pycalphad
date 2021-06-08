import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


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
    Extension('pycalphad.core.minimizer', sources=['pycalphad/core/minimizer.pyx']),
]

setup(
    name='pycalphad',
    author='Richard Otis',
    author_email='richard.otis@outlook.com',
    description='CALPHAD tools for designing thermodynamic models, calculating phase diagrams and investigating phase equilibria.',
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
    license='MIT',
    long_description=read('README.rst'),
    long_description_content_type='text/x-rst',
    url='https://pycalphad.org/',
    install_requires=[
        # NOTE: please try to keep any depedencies in alphabetic order so they
        # may be easily compared with other dependency lists
        # NOTE: these dependencies may differ in name from those in the
        # conda-forge Anaconda channel. For example, conda-forge/symengine
        # gives the C++ SymEngine library, while conda-forge/python-symengine
        # provides the Python package called `symengine`.
        'Cython>=0.24',
        'importlib_metadata',  # drop when pycalphad drops support for Python<3.8
        'matplotlib>=3.3',
        'numpy>=1.13',
        'pyparsing>=2.4',
        'pytest',
        'pytest-cov',
        'scipy',
        'setuptools_scm[toml]>=6.0',
        'symengine==0.7.2',  # python-symengine on conda-forge
        'sympy==1.8',
        'tinydb>=3.8',
        'xarray>=0.11.2',
    ],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Supported Python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

)
