from setuptools import setup
import os

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='pycalphad',
    version='0.0.1',
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
