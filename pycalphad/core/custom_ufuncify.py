"""
This module contains a modified version of the sympy function ufuncify.
We vendor the modified version until the patches make their way upstream.
"""
from __future__ import print_function, division
import sys
import os
import subprocess
import time
from string import Template
from .custom_codegen import CCodeGen
from .tempfilemanager import TempfileManager
from sympy.core.symbol import Symbol
from sympy.utilities.codegen import make_routine, OutputArgument, InOutArgument
from sympy.utilities.autowrap import CodeWrapper

#################################################################
#                           UFUNCIFY                            #
#################################################################

_ufunc_top = Template("""\
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
${include_files}

static PyMethodDef ${module}Methods[] = {
        {NULL, NULL, 0, NULL}
};""")

_ufunc_outcalls = Template("*((double *)out${outnum}) = ${funcname}(${call_args});")

_ufunc_body = Template("""\
static void ${funcname}_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    ${declare_args}
    ${declare_steps}
    for (i = 0; i < n; i++) {
        ${outcalls}
        ${step_increments}
    }
}
PyUFuncGenericFunction ${funcname}_funcs[1] = {&${funcname}_ufunc};
static char ${funcname}_types[${n_types}] = ${types}
static void *${funcname}_data[1] = {NULL};""")

_ufunc_bottom = Template("""\
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "${module}",
    NULL,
    -1,
    ${module}Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_${module}(void)
{
    PyObject *m, *d;
    ${function_creation}
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ${ufunc_init}
    return m;
}
#else
PyMODINIT_FUNC init${module}(void)
{
    PyObject *m, *d;
    ${function_creation}
    m = Py_InitModule("${module}", ${module}Methods);
    if (m == NULL) {
        return;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ${ufunc_init}
}
#endif\
""")

_ufunc_init_form = Template("""\
ufunc${ind} = PyUFunc_FromFuncAndData(${funcname}_funcs, ${funcname}_data, ${funcname}_types, 1, ${n_in}, ${n_out},
            PyUFunc_None, "${module}", ${docstring}, 0);
    PyDict_SetItemString(d, "${funcname}", ufunc${ind});
    Py_DECREF(ufunc${ind});""")

_ufunc_setup = Template("""\
def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('',
                           parent_package,
                           top_path)
    config.add_extension('${module}', sources=${sources},
                         extra_compile_args=${cflags})

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)""")

class UfuncifyCodeWrapper(CodeWrapper):
    """Wrapper for Ufuncify"""

    def __init__(self, generator, tmpman, filepath=None, flags=None, verbose=False):
        """
        generator -- the code generator to use
        """
        super(UfuncifyCodeWrapper, self).__init__(generator, filepath, flags, verbose)
        self.tmpman = tmpman

    @property
    def command(self):
        command = [sys.executable, "setup.py", "build_ext", "--inplace"]
        return command

    @property
    def module_name(self):
        return "%s_%s" % (self._module_basename, self._wrapped_hash)

    def wrap_code(self, routines, helpers=None, cflags=None):
        # This routine overrides CodeWrapper because we can't assume funcname == routines[0].name
        # Therefore we have to break the CodeWrapper private API.
        # There isn't an obvious way to extend multi-expr support to
        # the other autowrap backends, so we limit this change to ufuncify.
        helpers = helpers if helpers is not None else []
        cflags = cflags if cflags is not None else []
        # We just need a consistent name
        funcname = 'wrapped_' + str(id(routines) + id(helpers))
        self._wrapped_hash = id(routines) + id(helpers)
        # Need to save workdir so that it's properly pickled during multiprocessing
        self._workdir = self.tmpman.create_tree(self.filepath)
        #print('Creating pair ', (self._workdir, self.module_name))
        self._process = None
        self._module = None
        self._generate_code(routines, helpers)
        self._prepare_files(routines, funcname, cflags)
        # TODO: Should we move processing into a subprocess?
        self._process_files(routines)

        def lazy_wrapper(*args, **kwargs):
            if self._module is None:
                self._process.wait()
                # When using multiprocessing, wait() may return without actually being done
                # We may need to additionally let the import fail for some length of time
                if self._process.returncode != 0:
                    print('Error: Return code ', self._process.returncode)
                sys.path.append(self._workdir)
                # 30 second timeout
                timeout = time.time() + 30
                while True:
                    try:
                        mod = __import__(self.module_name)
                    except ImportError:
                        mod = None
                        #print('PROCESS PID:', self._process.pid)
                        #print('PROCESS RETURN CODE:', self._process.returncode)
                        if time.time() > timeout:
                            #print('FAILED TO IMPORT ', (self._workdir, self.module_name))
                            sys.path.remove(self._workdir)
                            raise ImportError((self._workdir, self.module_name))
                    except:
                        sys.path.remove(self._workdir)
                        raise
                    if mod is not None:
                        sys.path.remove(self._workdir)
                        break
                    time.sleep(1)
                self._module = mod
            return self._get_wrapped_function(self._module, funcname)(*args, **kwargs)

        return lazy_wrapper

    def _generate_code(self, main_routines, helper_routines):
        #print('GENERATE_CODE', self._workdir)
        all_routines = main_routines + helper_routines
        for routine in all_routines:
            #print(self._workdir, 'WRITING', self.filename + '_' + routine.name)
            self.generator.write(
                [routine], self.filename + '_' + routine.name, True, self.include_header,
                self.include_empty, cwd=self._workdir)

    def _prepare_files(self, routines, funcname, cflags):
        #print('PREPARE_FILES', self._workdir)
        #print(os.getcwd())
        # C
        codefilename = self.module_name + '.c'
        #print(self._workdir, 'WRITING', codefilename)
        with open(os.path.abspath(os.path.join(self._workdir, codefilename)), 'w') as f:
            self.dump_c(routines, f, self.filename, funcname=funcname)

        # setup.py
        with open(os.path.abspath(os.path.join(self._workdir, 'setup.py')), 'w') as f:
            self.dump_setup(f, routines, cflags=cflags)

    def _process_files(self, routine):
        command = self.command
        command.extend(self.flags)
        logfd = self.tmpman.create_logfile(prefix='ufuncify_', suffix='.log')
        #print(self._workdir, 'TMPMAN', self.tmpman.preserve_on_error)
        self._process = subprocess.Popen(command, stdout=logfd, stderr=logfd, cwd=self._workdir)
        #print(self._workdir, 'LAUNCH', self._process.pid)

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)

    def dump_setup(self, f, routines, cflags=None):
        cflags = cflags if cflags is not None else []
        sources = [self.module_name + '.c']
        sources.extend([self.filename + '_' + routine.name + '.c' for routine in routines])
        setup = _ufunc_setup.substitute(module=self.module_name,
                                        sources=str(sources),
                                        cflags=str(cflags))
        f.write(setup)

    def dump_c(self, routines, f, prefix, funcname=None):
        """Write a C file with python wrappers

        This file contains all the definitions of the routines in c code.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to name the imported module.
        funcname
            Name of the main function to be returned.
        """
        if (funcname is None) and (len(routines) == 1):
            funcname = routines[0].name
        elif funcname is None:
            raise ValueError('funcname must be specified for multiple output routines')
        functions = []
        function_creation = []
        ufunc_init = []
        module = self.module_name
        includes = [self.filename + '_' + routine.name for routine in routines]
        incl_directives = ['#include \"{0}.h\"'.format(i) for i in includes]
        include_files = "\n".join(incl_directives)
        top = _ufunc_top.substitute(include_files=include_files, module=module)

        name = funcname

        # Partition the C function arguments into categories
        # Here we assume all routines accept the same arguments
        r_index = 0
        py_in, _ = self._partition_args(routines[0].arguments)
        n_in = len(py_in)
        n_out = len(routines)

        # Declare Args
        form = "char *{0}{1} = args[{2}];"
        arg_decs = [form.format('in', i, i) for i in range(n_in)]
        arg_decs.extend([form.format('out', i, i+n_in) for i in range(n_out)])
        declare_args = '\n    '.join(arg_decs)

        # Declare Steps
        form = "npy_intp {0}{1}_step = steps[{2}];"
        step_decs = [form.format('in', i, i) for i in range(n_in)]
        step_decs.extend([form.format('out', i, i+n_in) for i in range(n_out)])
        declare_steps = '\n    '.join(step_decs)

        # Call Args
        form = "*(double *)in{0}"
        call_args = ', '.join([form.format(a) for a in range(n_in)])

        # Step Increments
        form = "{0}{1} += {0}{1}_step;"
        step_incs = [form.format('in', i) for i in range(n_in)]
        step_incs.extend([form.format('out', i, i) for i in range(n_out)])
        step_increments = '\n        '.join(step_incs)

        # Types
        n_types = n_in + n_out
        types = "{" + ', '.join(["NPY_DOUBLE"]*n_types) + "};"

        # Docstring
        docstring = '"Created in SymPy with Ufuncify"'

        # Function Creation
        function_creation.append("PyObject *ufunc{0};".format(r_index))

        # Ufunc initialization
        init_form = _ufunc_init_form.substitute(module=module,
                                                funcname=name,
                                                docstring=docstring,
                                                n_in=n_in, n_out=n_out,
                                                ind=r_index)
        ufunc_init.append(init_form)

        outcalls = [_ufunc_outcalls.substitute(outnum=i, call_args=call_args,
                                               funcname=routines[i].name) for i in range(n_out)]

        body = _ufunc_body.substitute(module=module, funcname=name,
                                      declare_args=declare_args,
                                      declare_steps=declare_steps,
                                      call_args=call_args,
                                      step_increments=step_increments,
                                      n_types=n_types, types=types, outcalls='\n        '.join(outcalls))
        functions.append(body)

        body = '\n\n'.join(functions)
        ufunc_init = '\n    '.join(ufunc_init)
        function_creation = '\n    '.join(function_creation)
        bottom = _ufunc_bottom.substitute(module=module,
                                          ufunc_init=ufunc_init,
                                          function_creation=function_creation)
        text = [top, body, bottom]
        f.write('\n\n'.join(text))

    def _partition_args(self, args):
        """Group function arguments into categories."""
        py_in = []
        py_out = []
        for arg in args:
            if isinstance(arg, OutputArgument):
                py_out.append(arg)
            elif isinstance(arg, InOutArgument):
                raise ValueError("Ufuncify doesn't support InOutArguments")
            else:
                py_in.append(arg)
        return py_in, py_out

@TempfileManager(os.getcwd())
def ufuncify(args, expr, tmpman=None, tempdir=None, flags=None, cflags=None, verbose=False, helpers=None):
    """Generates a binary function that supports broadcasting on numpy arrays.

    Parameters
    ----------
    args : iterable
        Either a Symbol or an iterable of symbols. Specifies the argument
        sequence for the function.
    expr : SymPy object or list of SymPy objects
        SymPy expression(s) that defines the element wise operation.
    tmpman : TempfileManager, optional
        Context manager for temporary file cleanup.
    tempdir : string, optional
        Path to directory for temporary files. If this argument is supplied,
        the generated code and the wrapper input files are left intact in the
        specified path.
    flags : iterable, optional
        Additional option flags that will be passed to the backend
    cflags : iterable, optional
        Additional compiler flags that will be passed to ``extra_compile_args``
    verbose : bool, optional
        If True, autowrap will not mute the command line backends. This can be
        helpful for debugging.
    helpers : iterable, optional
        Used to define auxillary expressions needed for the main expr. If the
        main expression needs to call a specialized function it should be put
        in the ``helpers`` iterable. Autowrap will then make sure that the
        compiled main expression can link to the helper routine. Items should
        be tuples with (<funtion_name>, <sympy_expression>, <arguments>). It
        is mandatory to supply an argument sequence to helper routines.
    """

    if isinstance(args, Symbol):
        args = (args,)
    else:
        args = tuple(args)
    if tmpman is None:
        raise ValueError('Missing temporary file context manager')

    helpers = helpers if helpers else ()
    flags = flags if flags else ()
    cflags = cflags if cflags else ()

    helps = []
    for name, expr, args in helpers:
        helps.append(make_routine(name, expr, args))
    code_wrapper = UfuncifyCodeWrapper(CCodeGen("ufuncify"), tmpman, tempdir,
                                       flags, verbose)
    if not isinstance(expr, (list, tuple)):
        expr = [expr]
    routines = [make_routine('autofunc{}'.format(idx), exprx, args) for idx, exprx in enumerate(expr)]
    return code_wrapper.wrap_code(routines, helpers=helps, cflags=cflags)