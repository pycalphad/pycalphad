import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


CYTHON_COMPILER_DIRECTIVES = {
    "language_level": 3,
}
CYTHON_DEFINE_MACROS = []
if os.getenv('CYTHON_COVERAGE', False):
    CYTHON_COMPILER_DIRECTIVES["linetrace"] = True
    CYTHON_DEFINE_MACROS.append(('CYTHON_TRACE_NOGIL', '1'))

CYTHON_EXTENSION_INCLUDES = ['.', np.get_include()]
CYTHON_EXTENSION_MODULES = [
    Extension('pycalphad.core.hyperplane', sources=['pycalphad/core/hyperplane.pyx'], define_macros=CYTHON_DEFINE_MACROS),
    Extension('pycalphad.core.eqsolver', sources=['pycalphad/core/eqsolver.pyx'], define_macros=CYTHON_DEFINE_MACROS),
    Extension('pycalphad.core.phase_rec', sources=['pycalphad/core/phase_rec.pyx'], define_macros=CYTHON_DEFINE_MACROS),
    Extension('pycalphad.core.composition_set', sources=['pycalphad/core/composition_set.pyx'], define_macros=CYTHON_DEFINE_MACROS),
    Extension('pycalphad.core.minimizer', sources=['pycalphad/core/minimizer.pyx'], define_macros=CYTHON_DEFINE_MACROS),
]

# https://cython.readthedocs.io/en/latest/src/tutorial/appendix.html
mingw32_link_args = [
    "-static-libgcc",
    "-static-libstdc++",
    "-Wl,-Bstatic,--whole-archive",
    "-lwinpthread",
    "-Wl,--no-whole-archive",
]

class Build(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "mingw32":
            for ext in self.extensions:
                ext.extra_link_args = mingw32_link_args
        return super().build_extensions()

setup(
    ext_modules=cythonize(
        CYTHON_EXTENSION_MODULES,
        include_path=CYTHON_EXTENSION_INCLUDES,
        compiler_directives=CYTHON_COMPILER_DIRECTIVES,
    ),
    cmdclass={"build_ext": Build},
    package_data={
        'pycalphad.core': ['*.pxd'] + (['*.pyx', '*.c', '*.h', '*.cpp', '*.hpp'] if os.getenv('CYTHON_COVERAGE', False) else []),
        'pycalphad.tests.databases': ['*'],
    },
    # This include is for the compiler to find the *.h files during the build_ext phase
    # the include must contain a symengine directory with header files
    include_dirs=[np.get_include()],
)
