"""
Tests for code generation from SymPy/SymEngine objects.
"""

from symengine.lib.symengine_wrapper import LambdaDouble, LLVMDouble
from pycalphad import Database, Model
from pycalphad.codegen.sympydiff_utils import build_functions, build_constraint_functions
from pycalphad.tests.datasets import ALNIPT_TDB


ALNIPT_DBF = Database(ALNIPT_TDB)


def test_build_functions_options():
    """The correct SymEngine backend can be chosen for build_functions"""
    mod = Model(ALNIPT_DBF, ['AL'], 'LIQUID')
    int_cons = mod.get_internal_constraints()

    backend = 'lambda'
    fs_lambda = build_functions(mod.GM, mod.GM.free_symbols,
                                include_obj=True, func_options={'backend': backend},
                                include_grad=True, grad_options={'backend': backend},
                                include_hess=True, hess_options={'backend': backend})
    assert isinstance(fs_lambda.func, LambdaDouble)
    assert isinstance(fs_lambda.grad, LambdaDouble)
    assert isinstance(fs_lambda.hess, LambdaDouble)

    cfs_lambda = build_constraint_functions(mod.GM.free_symbols, int_cons,
                                            func_options={'backend': backend},
                                            jac_options={'backend': backend},
                                            hess_options={'backend': backend})
    assert isinstance(cfs_lambda.cons_func, LambdaDouble)
    assert isinstance(cfs_lambda.cons_jac, LambdaDouble)
    assert isinstance(cfs_lambda.cons_hess, LambdaDouble)

    backend = 'llvm'
    fs_llvm = build_functions(mod.GM, mod.GM.free_symbols,
                              include_obj=True, func_options={'backend': backend},
                              include_grad=True, grad_options={'backend': backend},
                              include_hess=True, hess_options={'backend': backend})
    print(fs_llvm.func)
    print(fs_lambda.func)
    assert isinstance(fs_llvm.func, LLVMDouble)
    assert isinstance(fs_llvm.grad, LLVMDouble)
    assert isinstance(fs_llvm.hess, LLVMDouble)

    cfs_llvm = build_constraint_functions(mod.GM.free_symbols, int_cons,
                                          func_options={'backend': backend},
                                          jac_options={'backend': backend},
                                          hess_options={'backend': backend})
    assert isinstance(cfs_llvm.cons_func, LLVMDouble)
    assert isinstance(cfs_llvm.cons_jac, LLVMDouble)
    assert isinstance(cfs_llvm.cons_hess, LLVMDouble)
