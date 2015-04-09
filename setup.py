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
    packages=['pycalphad', 'pycalphad.eq', 'pycalphad.io', 'pycalphad.plot', 'pycalphad.plot.projections'],
    license='MIT',
    long_description=read('README.md'),
    url='https://github.com/richardotis/pycalphad',
    install_requires=['matplotlib', 'pandas', 'sympy', 'pyparsing', 'tinydb', 'scipy', 'numpy'],
    classifiers=['Development Status :: 3 - Alpha']
)
