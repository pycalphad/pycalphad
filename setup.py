import platform
from distutils.sysconfig import get_config_var
import sys
from distutils.version import LooseVersion
from setuptools import setup
import os
import versioneer

try:
    from Cython.Build import cythonize
    import numpy as np
    import scipy
except ImportError:
     raise ImportError("Cython, numpy and scipy must be installed before pycalphad can be installed.")

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

setup(
    name='pycalphad',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Richard Otis',
    author_email='richard.otis@outlook.com',
    description='CALPHAD tools for designing thermodynamic models, calculating phase diagrams and investigating phase equilibria.',
    packages=['pycalphad', 'pycalphad.codegen', 'pycalphad.core', 'pycalphad.io', 'pycalphad.plot'],
    # TODO: hardcoded include
    # The includes here are to pick up the *.pyx files
    # We might be able to get around these includes if symengine uses zip_safe=False
    # see: https://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html
    ext_modules=cythonize(['pycalphad/core/hyperplane.pyx', 'pycalphad/core/eqsolver.pyx',
                           'pycalphad/core/phase_rec.pyx',
                           'pycalphad/core/composition_set.pyx',
                           'pycalphad/core/problem.pyx'], include_path=['.', np.get_include(), '/Users/brandon/anaconda3/envs/calphad-dev/lib/python3.6/site-packages/symengine/lib']),

    package_data={
        'pycalphad/core': ['*.pxd'],
    },
    # TODO: hardcoded include
    # This include is for the compiler to find the *.h files during the build_ext phase
    include_dirs=[np.get_include(), '/Users/brandon/anaconda3/envs/calphad-dev/include/symengine/'],
    license='MIT',
    long_description=read('README.rst'),
    url='https://pycalphad.org/',
    install_requires=['matplotlib', 'pandas', 'xarray>=0.11.2', 'sympy==1.4', 'pyparsing', 'Cython>=0.24',
                      'tinydb>=3.8', 'scipy', 'numpy>=1.13', 'dask[complete]>=1.2', 'dill', 'ipopt'],
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
        'Programming Language :: Python :: 3.6'
        'Programming Language :: Python :: 3.7'
    ],

)
