"""
This file is only present in local installations to provide a mechanism for getting
the version from the source control management system.

It is excluded from MANIFEST.in and should only be present for local installations.

Inspired by:
https://stackoverflow.com/questions/43348746/how-to-detect-if-module-is-installed-in-editable-mode/66869528#66869528
"""

from setuptools_scm import get_version