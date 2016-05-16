"""
This module contains a patched version of sympy's CCodeGen to optimize the code printing.
"""

from sympy.utilities.codegen import CCodeGen as sympy_CCodeGen
from sympy.utilities.codegen import ResultBase, Result, AssignmentError
from sympy.printing import ccode

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