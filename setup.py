from setuptools import setup, Extension
import os
import versioneer

try:
    from Cython.Build import cythonize
    import numpy as np
    import scipy
except ImportError:
     raise ImportError("Cython, numpy, and scipy must be installed before pycalphad can be installed.")


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
    packages=['pycalphad', 'pycalphad.codegen', 'pycalphad.core', 'pycalphad.io', 'pycalphad.plot', 'pycalphad.plot.binary'],
    # "error: '::hypot' has not been declared when compiling with MingGW64"
    # https://github.com/Theano/Theano/issues/4926
    ext_modules=cythonize([Extension('pycalphad.core.hyperplane',
                                    sources=['pycalphad/core/hyperplane.pyx'],
                                    extra_compile_args=["-std=c++11", "-D_hypot=hypot"],extra_link_args=["-std=c++11"],
                                    include_dirs=['.', np.get_include()],
                                     ),
                           Extension('pycalphad.core.eqsolver',
                                    sources=['pycalphad/core/eqsolver.pyx'],
                                    extra_compile_args=["-std=c++11", "-D_hypot=hypot"],extra_link_args=["-std=c++11"],
                                    include_dirs=['.', np.get_include()],
                                    ),
                           Extension('pycalphad.core.phase_rec',
                                    sources=['pycalphad/core/phase_rec.pyx'],
                                    extra_compile_args=["-std=c++11", "-D_hypot=hypot"],
                                    include_dirs=['.', np.get_include()],
                                     ),
                           Extension('pycalphad.core.composition_set',
                                    sources=['pycalphad/core/composition_set.pyx'],
                                    extra_compile_args=["-std=c++11", "-D_hypot=hypot"],extra_link_args=["-std=c++11"],
                                    include_dirs=['.', np.get_include()],
                                     ),
                           Extension('pycalphad.core.problem',
                                    sources=['pycalphad/core/problem.pyx'],
                                    extra_compile_args=["-std=c++11", "-D_hypot=hypot"],extra_link_args=["-std=c++11"],
                                    include_dirs=['.', np.get_include()],
                                     ),

                          ], include_path=['.', np.get_include()]),
    package_data={
        'pycalphad/core': ['*.pxd'],
    },
    # This include is for the compiler to find the *.h files during the build_ext phase
    # the include must contain a symengine directory with header files
    # TODO: Brandon needed to add a CFLAGS='-std=c++11' before the setup.py build_ext command.
    include_dirs=[np.get_include()],
    license='MIT',
    long_description=read('README.rst'),
    url='https://pycalphad.org/',
    install_requires=[
        # NOTE: please try to keep any depedencies in alphabetic order so they
        # may be easily compared with other dependency lists
        # NOTE: these dependencies may differ in name from those in the
        # conda-forge Anaconda channel. For example, conda-forge/symengine
        # gives the C++ SymEngine library, while conda-forge/python-symengine
        # provides the Python package called `symengine`.
        'Cython>=0.24',
        'ipopt',
        'matplotlib',
        'numpy>=1.13',
        'pyparsing',
        'scipy',
        'symengine==0.6.1',
        'sympy==1.7',
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

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

)
