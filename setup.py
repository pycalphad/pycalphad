import platform
from distutils.sysconfig import get_config_var
import sys
from distutils.version import LooseVersion
from setuptools import setup, Extension
import os
import versioneer
from sysconfig import get_paths

try:
    from Cython.Build import cythonize
    import numpy as np
    import symengine
    import scipy
except ImportError:
     raise ImportError("Cython, numpy, SymEngine, and scipy must be installed before pycalphad can be installed.")


# These are related to a fix where Cython does not pick up the correct
# C++ standard library, due to changes in the macOS compiler toolchain
# See the changes here and discussion at the following links:
# https://github.com/pandas-dev/pandas/pull/24274 and
# https://github.com/pandas-dev/pandas/issues/23424#issuecomment-446393981
def is_platform_mac():
    return sys.platform == 'darwin'

# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
if is_platform_mac():
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(
            get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def symengine_pyx_get_include():
    return os.path.join(os.path.dirname(symengine.__file__), 'lib')


def symengine_lib_get_include():
    if sys.platform == 'win32':
        # Strictly only valid for recent conda installations
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(symengine.__file__)))), 'Library', 'lib')
    else:
        return os.path.join(os.path.dirname(symengine.__file__), 'lib')

def symengine_bin_get_include():
    if sys.platform == 'win32':
        # Strictly only valid for recent conda installations
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(symengine.__file__)))), 'Library', 'bin')
    else:
        return os.path.join(os.path.dirname(symengine.__file__), 'bin')

def symengine_h_get_include():
    if sys.platform == 'win32':
        # Strictly only valid for recent conda installations
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(symengine.__file__)))), 'Library', 'include')
    else:
        return os.path.dirname(get_paths()['include'])


setup(
    name='pycalphad',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Richard Otis',
    author_email='richard.otis@outlook.com',
    description='CALPHAD tools for designing thermodynamic models, calculating phase diagrams and investigating phase equilibria.',
    packages=['pycalphad', 'pycalphad.codegen', 'pycalphad.core', 'pycalphad.io', 'pycalphad.plot'],
    # The includes here are to pick up the *.pyx files
    # We might be able to get around these includes if symengine uses zip_safe=False
    # see: https://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html
    # "error: '::hypot' has not been declared when compiling with MingGW64"
    # https://github.com/Theano/Theano/issues/4926
    ext_modules=cythonize([Extension('pycalphad.core.hyperplane',
                                    sources=['pycalphad/core/hyperplane.pyx'],
                                    extra_compile_args=["-std=c++11", "-D_hypot=hypot"],extra_link_args=["-std=c++11"],
                                    include_dirs=['.', np.get_include(), symengine_pyx_get_include(),'/home/rotis/git/symengine/build/include'],
                                    library_dirs=[symengine_lib_get_include(),symengine_bin_get_include(),'/home/rotis/git/symengine/build/lib64'],
                                    libraries=['symengine']
                                     ),
                           Extension('pycalphad.core.eqsolver',
                                    sources=['pycalphad/core/eqsolver.pyx'],
                                    extra_compile_args=["-std=c++11", "-D_hypot=hypot"],extra_link_args=["-std=c++11"],
                                    include_dirs=['.', np.get_include(), symengine_pyx_get_include(),'/home/rotis/git/symengine/build/include'],
                                    library_dirs=[symengine_lib_get_include(),symengine_bin_get_include(),'/home/rotis/git/symengine/build/lib64'],
                                    libraries=['symengine']
                                    ),
                           Extension('pycalphad.core.phase_rec',
                                    sources=['pycalphad/core/phase_rec.pyx'],
                                    extra_compile_args=["-std=c++11", "-D_hypot=hypot"],
                                    include_dirs=['.', np.get_include(), symengine_pyx_get_include(), '/home/rotis/git/symengine/build/include'],
                                    library_dirs=[symengine_lib_get_include(),symengine_bin_get_include(),'/home/rotis/git/symengine/build/lib64'],
                                    extra_link_args=['-std=c++11'],
                                    libraries=['symengine']
                                     ),
                           Extension('pycalphad.core.composition_set',
                                    sources=['pycalphad/core/composition_set.pyx'],
                                    extra_compile_args=["-std=c++11", "-D_hypot=hypot"],extra_link_args=["-std=c++11"],
                                    include_dirs=['.', np.get_include(), symengine_pyx_get_include(), '/home/rotis/git/symengine/build/include'],
                                    library_dirs=[symengine_lib_get_include(),symengine_bin_get_include(),'/home/rotis/git/symengine/build/lib64'],
                                    libraries=['symengine']
                                     ),
                           Extension('pycalphad.core.problem',
                                    sources=['pycalphad/core/problem.pyx'],
                                    extra_compile_args=["-std=c++11", "-D_hypot=hypot"],extra_link_args=["-std=c++11"],
                                    include_dirs=['.', np.get_include(), symengine_pyx_get_include(), '/home/rotis/git/symengine/build/include'],
                                    library_dirs=[symengine_lib_get_include(),symengine_bin_get_include(),'/home/rotis/git/symengine/build/lib64'],
                                    libraries=['symengine']
                                     ),

                          ], include_path=['.', np.get_include(), symengine_pyx_get_include(), symengine_lib_get_include(),symengine_bin_get_include(),'/home/rotis/git/symengine/build/include']),
    package_data={
        'pycalphad/core': ['*.pxd'],
    },
    # This include is for the compiler to find the *.h files during the build_ext phase
    # the include must contain a symengine directory with header files
    # TODO: Brandon needed to add a CFLAGS='-std=c++11' before the setup.py build_ext command.
    include_dirs=[np.get_include(), symengine_pyx_get_include(), symengine_h_get_include()],
    license='MIT',
    long_description=read('README.rst'),
    url='https://pycalphad.org/',
    install_requires=['matplotlib', 'pandas', 'xarray>=0.11.2', 'sympy==1.4', 'pyparsing', 'Cython>=0.24',
                      'tinydb>=3.8', 'scipy', 'numpy>=1.13', 'ipopt', 'symengine'],
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],

)
