from setuptools import setup
import os
import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = 'pycalphad/_version.py'
versioneer.versionfile_build = 'pycalphad/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = 'pycalphad-' # dirname like 'myproject-1.2.0'

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
    license='MIT',
    long_description=read('README.rst'),
    url='https://github.com/richardotis/pycalphad',
    install_requires=['matplotlib', 'pandas', 'xarray', 'sympy', 'pyparsing', 'tqdm',
                      'autograd', 'tinydb', 'scipy', 'numpy>=1.9'],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

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
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],

)
