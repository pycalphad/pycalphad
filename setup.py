from setuptools import setup
import os
import versioneer

try:
    from Cython.Build import cythonize
    import numpy as np
    import scipy
except ImportError:
     raise ImportError("Cython, numpy and scipy must be installed before pycalphad can be installed.")

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='pycalphad',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Richard Otis',
    author_email='richard.otis@outlook.com',
    description='CALPHAD tools for designing thermodynamic models, calculating phase diagrams and investigating phase equilibria.',
    packages=['pycalphad', 'pycalphad.core', 'pycalphad.io', 'pycalphad.plot'],
    ext_modules=cythonize(['pycalphad/core/hyperplane.pyx', 'pycalphad/core/eqsolver.pyx',
                           'pycalphad/core/phase_rec.pyx',
                           'pycalphad/core/composition_set.pyx',
                           'pycalphad/core/problem.pyx']),
    package_data={
        'pycalphad/core': ['*.pxd'],
    },
    include_dirs=[np.get_include()],
    license='MIT',
    long_description=read('README.rst'),
    url='https://pycalphad.org/',
    install_requires=['matplotlib', 'pandas', 'xarray!=0.8', 'sympy', 'pyparsing', 'Cython>=0.24',
                      'tinydb', 'scipy', 'numpy>=1.9', 'dask[complete]>=0.15', 'dill', 'ipopt'],
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

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],

)
