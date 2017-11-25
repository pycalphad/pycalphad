"""Module for compiling codegen output, and wrap the binary for use in
python.

.. note:: To use the autowrap module it must first be imported

   >>> from sympy.utilities.autowrap import autowrap

This module provides a common interface for different external backends, such
as f2py, fwrap, Cython, SWIG(?) etc. (Currently only f2py and Cython are
implemented) The goal is to provide access to compiled binaries of acceptable
performance with a one-button user interface, i.e.

    >>> from sympy.abc import x,y
    >>> expr = ((x - y)**(25)).expand()
    >>> binary_callable = autowrap(expr)
    >>> binary_callable(1, 2)
    -1.0

The callable returned from autowrap() is a binary python function, not a
SymPy object.  If it is desired to use the compiled function in symbolic
expressions, it is better to use binary_function() which returns a SymPy
Function object.  The binary callable is attached as the _imp_ attribute and
invoked when a numerical evaluation is requested with evalf(), or with
lambdify().

    >>> from sympy.utilities.autowrap import binary_function
    >>> f = binary_function('f', expr)
    >>> 2*f(x, y) + y
    y + 2*f(x, y)
    >>> (2*f(x, y) + y).evalf(2, subs={x: 1, y:2})
    0.e-110

The idea is that a SymPy user will primarily be interested in working with
mathematical expressions, and should not have to learn details about wrapping
tools in order to evaluate expressions numerically, even if they are
computationally expensive.

When is this useful?

    1) For computations on large arrays, Python iterations may be too slow,
       and depending on the mathematical expression, it may be difficult to
       exploit the advanced index operations provided by NumPy.

    2) For *really* long expressions that will be called repeatedly, the
       compiled binary should be significantly faster than SymPy's .evalf()

    3) If you are generating code with the codegen utility in order to use
       it in another project, the automatic python wrappers let you test the
       binaries immediately from within SymPy.

    4) To create customized ufuncs for use with numpy arrays.
       See *ufuncify*.

When is this module NOT the best approach?

    1) If you are really concerned about speed or memory optimizations,
       you will probably get better results by working directly with the
       wrapper tools and the low level code.  However, the files generated
       by this utility may provide a useful starting point and reference
       code. Temporary files will be left intact if you supply the keyword
       tempdir="path/to/files/".

    2) If the array computation can be handled easily by numpy, and you
       don't need the binaries for another project.

"""
from __future__ import print_function
from sympy.core.compatibility import iterable
try:
    from sympy.utilities.codegen import CCodePrinter
except ImportError:
    from sympy.printing.ccode import C89CodePrinter as CCodePrinter
from sympy.utilities.codegen import (AssignmentError, OutputArgument, ResultBase,
                                     Result, CodeGenArgumentListError,
                                     CCodeGen, Variable)
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.autowrap import CythonCodeWrapper, DummyWrapper
from subprocess import STDOUT, CalledProcessError, check_output
from sympy.printing.ccode import ccode
import os
import sys
import tempfile
import uuid


# Here we define a lookup of backends -> tuples of languages. For now, each
# tuple is of length 1, but if a backend supports more than one language,
# the most preferable language is listed first.
_lang_lookup = {'CYTHON': ('C',),
                'DUMMY': ('F95',)}     # Dummy here just for testing


def _infer_language(backend):
    """For a given backend, return the top choice of language"""
    langs = _lang_lookup.get(backend.upper(), False)
    if not langs:
        raise ValueError("Unrecognized backend: " + backend)
    return langs[0]


def import_extension(path, modname):
    import glob
    npath = glob.glob(os.path.join(path, modname+'.*'))
    # Blacklist fixes gh-65.
    # We filter out any files that can be created by compilers which are not our actual compiled file.
    # We cannot more directly search for our files because of differing platforms.
    blacklist = ['dSYM', 'c', 'pyx']
    if os.name == 'nt':
        blacklist.extend(['def', 'o'])
    npath = [x for x in npath if x.split('.')[-1] not in blacklist]
    if len(npath) == 1:
        npath = npath[0]
    else:
        raise ImportError('Failed to import', os.path.join(path, modname+'.*'), ' len(npath)=', len(npath))
    try:
        # Python 3.5+
        import importlib.util
        spec = importlib.util.spec_from_file_location(modname, npath)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
    except (AttributeError, ImportError):
        try:
            # Python 3.4
            from importlib.machinery import ExtensionFileLoader
            foo = ExtensionFileLoader(modname, npath).load_module()
        except ImportError:
            # Python 2.7
            import imp
            foo = imp.load_dynamic(modname, npath)
    return foo

class CodeWrapError(Exception):
    pass

class C89CodePrinter(CCodePrinter):
    """
    C89-compatible code printing allows for Windows compatibility.
    (MSVC 14 and newer support C99, but we are going for broad compatibility.)
    """
    def _get_loop_opening_ending(self, indices):
        # The purpose is to enable C89-compliant loops (indices are pre-declared)
        open_lines = []
        close_lines = []
        loopstart = "for (%(var)s=%(start)s; %(var)s<%(end)s; %(var)s++){"
        for i in indices:
            # C arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'var': self._print(i.label),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines

class CustomCCodeGen(CCodeGen):
    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: http://en.wikipedia.org/wiki/Function_prototype

        """
        if len(routine.results) == 1:
            ctype = routine.results[0].get_datatype('C')
        else:
            ctype = "void"

        type_args = []
        for arg in routine.arguments:
            name = ccode(arg.name)
            # Hack to make all double-valued arguments into pointers
            if arg.dimensions or isinstance(arg, ResultBase) or arg.get_datatype('C') == 'double':
                type_args.append((arg.get_datatype('C'), "*%s" % name))
            else:
                type_args.append((arg.get_datatype('C'), name))
        arguments = ", ".join([ "%s %s" % t for t in type_args])
        return "%s %s(%s)" % (ctype, routine.name, arguments)

    def _declare_locals(self, routine):
        # loop variables are declared at the top to enable C89 support
        retlines = []
        for lcv in routine.local_vars:
            t = Variable(lcv).get_datatype('c')
            retlines.append('{0} {1};\n'.format(t, lcv.name))
        return retlines

    def _call_printer(self, routine):
        code_lines = []

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        return_val = None
        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = routine.name + "_result"
                t = result.get_datatype('c')
                code_lines.append("{0} {1};\n".format(t, str(assign_to)))
                return_val = assign_to
            else:
                assign_to = result.result_var

            try:
                constants, not_c, c_expr = C89CodePrinter({'human': False, 'dereference': dereference}).doprint(
                    result.expr, assign_to)
            except AssignmentError:
                assign_to = result.result_var
                code_lines.append(
                    "%s %s;\n" % (result.get_datatype('c'), str(assign_to)))
                constants, not_c, c_expr = C89CodePrinter({'human': False, 'dereference': dereference}).doprint(
                    result.expr, assign_to)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))
            code_lines.append("%s\n" % c_expr)

        if return_val:
            code_lines.append("   return %s;\n" % return_val)
        return code_lines

    def dump_h(self, routines, f, prefix, header=True, empty=True):
        """Writes the C header file.

        This file contains all the function declarations.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to construct the include guards.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """
        if header:
            print(''.join(self._get_header()), file=f)
        guard_name = "%s__%s__H" % (self.project.replace(" ", "_").upper(),
                                    prefix.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_").replace("-", "_").upper())
        # include guards
        if empty:
            print(file=f)
        print("#ifndef %s" % guard_name, file=f)
        print("#define %s" % guard_name, file=f)
        if empty:
            print(file=f)
        # declaration of the function prototypes
        for routine in routines:
            prototype = self.get_prototype(routine)
            print("%s;" % prototype, file=f)
        # end if include guards
        if empty:
            print(file=f)
        print("#endif", file=f)
        if empty:
            print(file=f)
    dump_h.extension = CCodeGen.interface_extension

    dump_fns = [CCodeGen.dump_c, dump_h]


class ThreadSafeCythonCodeWrapper(CythonCodeWrapper):
    setup_template = (
        "from distutils.core import setup\n"
        "from distutils.extension import Extension\n"
        "from Cython.Distutils import build_ext\n"
        "{np_import}"
        "\n"
        "setup(\n"
        "    cmdclass = {{'build_ext': build_ext}},\n"
        "    ext_modules = [Extension({ext_args},\n"
        "                             extra_compile_args=['-std=c99'])],\n"
        "{np_includes}"
        "        )")

    pyx_imports = (
        "import numpy as np\n"
        "cimport numpy as np\n"
        "from cpython cimport PyCapsule_New\n\n")

    pyx_header = (
        "cdef extern from '{header_file}.h':\n"
        "    {prototype}\n\n")

    pyx_func = (
        "def {name}_c({arg_string}):\n"
        "\n"
        "{declarations}"
        "{body}\n"
        "def get_pointer_c():\n"
        "    return PyCapsule_New(<void*>{name}, NULL, NULL)\n")

    def __init__(self, *args, **kwargs):
        super(ThreadSafeCythonCodeWrapper, self).__init__(*args, **kwargs)
        self._module_id = str(uuid.uuid4()).replace('-', '_')
        self.filepath = self.filepath or tempfile.mkdtemp("_sympy_compile")

    @property
    def filename(self):
        return "%s_%s" % (self._filename, self._module_id)

    @property
    def command(self):
        command = [sys.executable, os.path.join(self.filepath, "setup.py"), "build_ext", "--build-lib", self.filepath]
        return command

    @property
    def module_name(self):
        return "%s_%s" % (self._module_basename, self._module_id)

    def _prepare_files(self, routine):
        pyxfilename = self.module_name + '.pyx'
        codefilename = "%s.%s" % (self.filename, self.generator.code_extension)

        # pyx
        with open(os.path.join(self.filepath, pyxfilename), 'w') as f:
            self.dump_pyx([routine], f, str(os.path.join(self.filepath, self.filename)).replace(os.sep, '/'))

        # setup.py
        ext_args = [repr(self.module_name), repr([os.path.join(self.filepath, pyxfilename),
                                                  os.path.join(self.filepath, codefilename)])]
        if self._need_numpy:
            np_import = 'import numpy as np\n'
            np_includes = '    include_dirs = [np.get_include()],\n'
        else:
            np_import = ''
            np_includes = ''
        with open(os.path.join(self.filepath, 'setup.py'), 'w') as f:
            f.write(self.setup_template.format(ext_args=", ".join(ext_args),
                                               np_import=np_import,
                                               np_includes=np_includes))

    def _process_files(self, routine):
        command = self.command
        command.extend(self.flags)
        try:
            retoutput = check_output(command, stderr=STDOUT)
        except CalledProcessError as e:
            raise CodeWrapError(
                "Error while executing command: %s. Command output is:\n%s" % (
                    " ".join(command), e.output.decode()))

    def _prototype_arg(self, arg):
        mat_dec = "np.ndarray[{mtype}, ndim={ndim}] {name}"
        np_types = {'double': 'np.double_t',
                    'int': 'np.int_t'}
        t = arg.get_datatype('c')
        # Hack to force all doubles to be at least 1-D arrays
        if arg.dimensions or t == 'double':
            self._need_numpy = True
            if arg.dimensions:
                ndim = len(arg.dimensions)
            else:
                ndim = 1
            mtype = np_types[t]
            return mat_dec.format(mtype=mtype, ndim=ndim, name=arg.name)
        else:
            return "%s %s" % (t, str(arg.name))

    def _call_arg(self, arg):
        t = arg.get_datatype('c')
        if arg.dimensions or t == 'double':
            return "<{0}*> {1}.data".format(t, arg.name)
        elif isinstance(arg, ResultBase):
            return "&{0}".format(arg.name)
        else:
            return str(arg.name)

    def wrap_code(self, routine, helpers=None):
        helpers = helpers if helpers is not None else []
        workdir = self.filepath
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        try:
            self.generator.write(
                [routine]+helpers, str(os.path.join(workdir, self.filename)).replace(os.sep, '/'), True, self.include_header,
                self.include_empty)
            self._prepare_files(routine)
            self._process_files(routine)
            mod = import_extension(workdir, self.module_name)
        finally:
            self._module_id = str(uuid.uuid4()).replace('-', '_')
        return self._get_wrapped_function(mod, routine.name), self._get_wrapped_function(mod, 'get_pointer')(), str(self.module_name), str(routine.name)


def _get_code_wrapper_class(backend):
    wrappers = {'CYTHON': ThreadSafeCythonCodeWrapper,
        'DUMMY': DummyWrapper}
    return wrappers[backend.upper()]


def _validate_backend_language(backend, language):
    """Throws error if backend and language are incompatible"""
    langs = _lang_lookup.get(backend.upper(), False)
    if not langs:
        raise ValueError("Unrecognized backend: " + backend)
    if language.upper() not in langs:
        raise ValueError(("Backend {0} and language {1} are "
                          "incompatible").format(backend, language))


def get_code_generator(language, project):
    CodeGenClass = {"C": CustomCCodeGen}.get(language.upper())
    if CodeGenClass is None:
        raise ValueError("Language '%s' is not supported." % language)
    return CodeGenClass(project)


def make_routine(name, expr, argument_sequence=None,
                 global_vars=None, language="C"):
    """A factory that makes an appropriate Routine from an expression.

    Parameters
    ==========

    name : string
        The name of this routine in the generated code.

    expr : expression or list/tuple of expressions
        A SymPy expression that the Routine instance will represent.  If
        given a list or tuple of expressions, the routine will be
        considered to have multiple return values and/or output arguments.

    argument_sequence : list or tuple, optional
        List arguments for the routine in a preferred order.  If omitted,
        the results are language dependent, for example, alphabetical order
        or in the same order as the given expressions.

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    language : string, optional
        Specify a target language.  The Routine itself should be
        language-agnostic but the precise way one is created, error
        checking, etc depend on the language.  [default: "F95"].

    A decision about whether to use output arguments or return values is made
    depending on both the language and the particular mathematical expressions.
    For an expression of type Equality, the left hand side is typically made
    into an OutputArgument (or perhaps an InOutArgument if appropriate).
    Otherwise, typically, the calculated expression is made a return values of
    the routine.

    Examples
    ========

    >>> from sympy.utilities.codegen import make_routine
    >>> from sympy.abc import x, y, f, g
    >>> from sympy import Eq
    >>> r = make_routine('test', [Eq(f, 2*x), Eq(g, x + y)])
    >>> [arg.result_var for arg in r.results]
    []
    >>> [arg.name for arg in r.arguments]
    [x, y, f, g]
    >>> [arg.name for arg in r.result_variables]
    [f, g]
    >>> r.local_vars
    set()

    Another more complicated example with a mixture of specified and
    automatically-assigned names.  Also has Matrix output.

    >>> from sympy import Matrix
    >>> r = make_routine('fcn', [x*y, Eq(f, 1), Eq(g, x + g), Matrix([[x, 2]])])
    >>> [arg.result_var for arg in r.results]  # doctest: +SKIP
    [result_5397460570204848505]
    >>> [arg.expr for arg in r.results]
    [x*y]
    >>> [arg.name for arg in r.arguments]  # doctest: +SKIP
    [x, y, f, g, out_8598435338387848786]

    We can examine the various arguments more closely:

    >>> from sympy.utilities.codegen import (InputArgument, OutputArgument,
    ...                                      InOutArgument)
    >>> [a.name for a in r.arguments if isinstance(a, InputArgument)]
    [x, y]

    >>> [a.name for a in r.arguments if isinstance(a, OutputArgument)]  # doctest: +SKIP
    [f, out_8598435338387848786]
    >>> [a.expr for a in r.arguments if isinstance(a, OutputArgument)]
    [1, Matrix([[x, 2]])]

    >>> [a.name for a in r.arguments if isinstance(a, InOutArgument)]
    [g]
    >>> [a.expr for a in r.arguments if isinstance(a, InOutArgument)]
    [g + x]

    """

    # initialize a new code generator
    code_gen = get_code_generator(language, "nothingElseMatters")

    return code_gen.routine(name, expr, argument_sequence, global_vars)


def autowrap(
    expr, language=None, backend='Cython', tempdir=None, args=None, flags=None,
    verbose=False, helpers=None):
    """Generates python callable binaries based on the math expression.

    Parameters
    ----------
    expr
        The SymPy expression that should be wrapped as a binary routine.
    language : string, optional
        If supplied, (options: 'C' or 'F95'), specifies the language of the
        generated code. If ``None`` [default], the language is inferred based
        upon the specified backend.
    backend : string, optional
        Backend used to wrap the generated code. Either 'f2py',
        or 'cython' [default].
    tempdir : string, optional
        Path to directory for temporary files. If this argument is supplied,
        the generated code and the wrapper input files are left intact in the
        specified path.
    args : iterable, optional
        An ordered iterable of symbols. Specifies the argument sequence for the
        function.
    flags : iterable, optional
        Additional option flags that will be passed to the backend.
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

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.autowrap import autowrap
    >>> expr = ((x - y + z)**(13)).expand()
    >>> binary_func = autowrap(expr)
    >>> binary_func(1, 4, 2)
    -1.0
    """
    if language:
        _validate_backend_language(backend, language)
    else:
        language = _infer_language(backend)

    helpers = [helpers] if helpers else ()
    flags = flags if flags else ()
    args = list(args) if iterable(args, exclude=set) else args

    code_generator = get_code_generator(language, "autowrap")
    CodeWrapperClass = _get_code_wrapper_class(backend)
    code_wrapper = CodeWrapperClass(code_generator, tempdir, flags, verbose)

    helps = []
    for name_h, expr_h, args_h in helpers:
        helps.append(make_routine(name_h, expr_h, args_h))

    for name_h, expr_h, args_h in helpers:
        if expr.has(expr_h):
            name_h = binary_function(name_h, expr_h, backend = 'dummy')
            expr = expr.subs(expr_h, name_h(*args_h))
    try:
        routine = make_routine('autofunc', expr, args)
    except CodeGenArgumentListError as e:
        # if all missing arguments are for pure output, we simply attach them
        # at the end and try again, because the wrappers will silently convert
        # them to return values anyway.
        new_args = []
        for missing in e.missing_args:
            if not isinstance(missing, OutputArgument):
                raise
            new_args.append(missing.name)
        routine = make_routine('autofunc', expr, args + new_args)

    return code_wrapper.wrap_code(routine, helpers=helps)


def binary_function(symfunc, expr, **kwargs):
    """Returns a sympy function with expr as binary implementation

    This is a convenience function that automates the steps needed to
    autowrap the SymPy expression and attaching it to a Function object
    with implemented_function().

    >>> from sympy.abc import x, y
    >>> from sympy.utilities.autowrap import binary_function
    >>> expr = ((x - y)**(25)).expand()
    >>> f = binary_function('f', expr)
    >>> type(f)
    <class 'sympy.core.function.UndefinedFunction'>
    >>> 2*f(x, y)
    2*f(x, y)
    >>> f(x, y).evalf(2, subs={x: 1, y: 2})
    -1.0
    """
    binary = autowrap(expr, **kwargs)
    return implemented_function(symfunc, binary)
