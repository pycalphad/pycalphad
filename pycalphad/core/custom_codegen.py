"""
This module contains a patched version of sympy's CCodeGen to optimize the code printing.
"""

from sympy.utilities.codegen import CCodeGen as sympy_CCodeGen
from sympy.utilities.codegen import FCodeGen as sympy_FCodeGen
from sympy.utilities.codegen import ResultBase, Result, AssignmentError,\
    InOutArgument, OutputArgument, get_default_datatype
from sympy.printing.codeprinter import Assignment
from sympy.printing import ccode
from sympy.utilities.codegen import FCodePrinter as sympy_FCodePrinter
from sympy import Function
try:
    # Python 2
    from StringIO import StringIO
except ImportError:
    # Python 3
    from io import StringIO
import os


class FCodePrinter(sympy_FCodePrinter):
    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assignment):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) then" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                else:
                    lines.append("else if (%s) then" % self._print(c))
                lines.append(self._print(e))
            lines.append("end if")
            return "\n".join(lines)
        elif self._settings["standard"] >= 95:
            # Only supported in F95 and newer:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            pattern = "merge(DBLE({T}), DBLE({F}), {COND})"
            code = self._print(expr.args[-1].expr)
            terms = list(expr.args[:-1])
            while terms:
                e, c = terms.pop()
                expr = self._print(e)
                cond = self._print(c)
                code = pattern.format(T=expr, F=code, COND=cond)
            return code
        else:
            # `merge` is not supported prior to F95
            raise NotImplementedError("Using Piecewise as an expression using "
                                      "inline operators is not supported in "
                                      "standards earlier than Fortran95.")

class FCodeGen(sympy_FCodeGen):
    """Generator for Fortran 95 code

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.f90 and <prefix>.h respectively.

    """

    def _get_symbol(self, s):
        """Returns the symbol as fcode prints it."""
        return FCodePrinter().doprint(s).strip()

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = routine.name
            elif isinstance(result, (OutputArgument, InOutArgument)):
                assign_to = result.result_var
            else:
                assign_to = None

            constants, not_fortran, f_expr = FCodePrinter({'source_format': 'free',
                                                           'human': False,
                                                           'standard': 95}).doprint(result.expr, assign_to=assign_to)

            for obj, v in sorted(constants, key=str):
                t = get_default_datatype(obj)
                declarations.append(
                    "%s, parameter :: %s = %s\n" % (t.fname, obj, v))
            for obj in sorted(not_fortran, key=str):
                t = get_default_datatype(obj)
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append("%s :: %s\n" % (t.fname, name))

            code_lines.append("%s\n" % f_expr)
        return declarations + code_lines

    def _indent_code(self, codelines):
        p = FCodePrinter({'source_format': 'free', 'human': False})
        return p.indent_code(codelines)


class CCodeGen(sympy_CCodeGen):
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
                # order='none' is an optimization not in upstream
                constants, not_c, c_expr = ccode(result.expr, human=False,
                                                 assign_to=assign_to, dereference=dereference, order='none')
            except AssignmentError:
                assign_to = result.result_var
                code_lines.append(
                    "%s %s;\n" % (result.get_datatype('c'), str(assign_to)))
                constants, not_c, c_expr = ccode(result.expr, human=False,
                                                 assign_to=assign_to, dereference=dereference, order='none')

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))
            code_lines.append("%s\n" % c_expr)

        if return_val:
            code_lines.append("   return %s;\n" % return_val)
        return code_lines

    def write(self, routines, prefix, to_files=False, header=True, empty=True, cwd=None):
        """Writes all the source code files for the given routines.

        The generated source is returned as a list of (filename, contents)
        tuples, or is written to files (see below).  Each filename consists
        of the given prefix, appended with an appropriate extension.

        Parameters
        ==========

        routines : list
            A list of Routine instances to be written

        prefix : string
            The prefix for the output files

        to_files : bool, optional
            When True, the output is written to files.  Otherwise, a list
            of (filename, contents) tuples is returned.  [default: False]

        header : bool, optional
            When True, a header comment is included on top of each source
            file. [default: True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files. [default: True]

        cwd : str, optional
            Working directory for writing files. [default: os.getcwd()]
        """
        cwd = os.getcwd() if cwd is None else cwd
        if to_files:
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                with open(os.path.abspath(os.path.join(cwd, filename)), "w") as f:
                    dump_fn(self, routines, f, prefix, header, empty)
        else:
            result = []
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                contents = StringIO()
                dump_fn(self, routines, contents, prefix, header, empty)
                result.append((filename, contents.getvalue()))
            return result