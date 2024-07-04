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
from functools import partial
import os

# https://github.com/pypa/setuptools_scm/issues/455#issuecomment-1268134018
# setuptools_scm config local_scheme via env var SETUPTOOLS_SCM_LOCAL_SCHEME
# Needed for TestPyPI uploads, to remove unsupported local version identifier
scm_local_scheme = "node-and-date"

if "SETUPTOOLS_SCM_LOCAL_SCHEME" in os.environ:
    local_scheme_values = [
        "node-and-date",
        "node-and-timestamp",
        "dirty-tag",
        "no-local-version",
    ]
    if os.environ["SETUPTOOLS_SCM_LOCAL_SCHEME"] in local_scheme_values:
        scm_local_scheme = os.environ["SETUPTOOLS_SCM_LOCAL_SCHEME"]
    else:
        raise ValueError(f'SETUPTOOLS_SCM_LOCAL_SCHEME is none of {local_scheme_values} - Got: {os.environ["SETUPTOOLS_SCM_LOCAL_SCHEME"]}')

get_version = partial(get_version, local_scheme=scm_local_scheme)