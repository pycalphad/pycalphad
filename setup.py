from setuptools import setup

setup(
    name='pycalphad',
    version='0.0.1',
    author='Richard Otis',
    author_email='richard.otis@outlook.com',
    packages=['calphad','calphad.io', 'calphad.minimize', 'calphad.plot','calphad.plot.projections'],
    license='',
    install_requires=['numpy','matplotlib','pandas', 'sympy','pyparsing', 'tinydb'],
    classifiers=['Development Status :: 3 - Alpha']
)
