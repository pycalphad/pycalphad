"""
This subpackage is only present in local installations. It is NOT distributed in source
or binary distributions.

It is excluded from MANIFEST.in, but it's also required to be in a subpackage because
bdists do not honor MANIFEST.in exclusions and the setup.py _must_ exclude this
manually. See: https://github.com/pypa/setuptools/issues/511

Inspired by:
https://stackoverflow.com/questions/43348746/how-to-detect-if-module-is-installed-in-editable-mode/66869528#66869528
"""

# Provide a mechanism for getting the version from the source control management system
from setuptools_scm import get_version