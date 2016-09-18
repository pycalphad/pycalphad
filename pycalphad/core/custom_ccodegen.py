"""
This module contains a patched version of sympy's CCodeGen to optimize the code printing.
"""

from sympy.utilities.codegen import CCodeGen as sympy_CCodeGen
from sympy.utilities.codegen import ResultBase, Result, AssignmentError
from sympy.printing import ccode
try:
    # Python 2
    from StringIO import StringIO
except ImportError:
    # Python 3
    from io import StringIO
import os

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