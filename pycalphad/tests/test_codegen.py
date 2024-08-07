"""
Tests for code generation from SymPy/SymEngine objects.
"""

import pickle
import pytest
import numpy as np
from symengine.lib.symengine_wrapper import LambdaDouble, LLVMDouble
from symengine import zoo
from pycalphad import Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.codegen.sympydiff_utils import build_functions, build_constraint_functions
from pycalphad.tests.fixtures import select_database, load_database


@select_database("alnipt.tdb")
def test_build_functions_options(load_database):
    """The correct SymEngine backend can be chosen for build_functions"""
    dbf = load_database()
    mod = Model(dbf, ['AL'], 'LIQUID')
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


@select_database("alnipt.tdb")
def test_phase_records_are_picklable(load_database):
    dbf = load_database()
    dof = np.array([300, 1.0])

    mod = Model(dbf, ['AL'], 'LIQUID')
    prxs = PhaseRecordFactory(dbf, [v.Species('AL')], [v.T], {'LIQUID': mod})
    prx_liquid = prxs['LIQUID']

    out = np.array([0.0])
    prx_liquid.obj(out, dof)

    prx_loaded = pickle.loads(pickle.dumps(prx_liquid))
    out_unpickled = np.array([0.0])
    prx_loaded.obj(out_unpickled, dof)

    assert np.isclose(out_unpickled[0], -1037.653911)
    assert np.all(out == out_unpickled)


@pytest.mark.xfail
@select_database("cfe_broshe.tdb")
def test_complex_infinity_can_build_callables_successfully(load_database):
    """Test that functions that containing complex infinity can be built with codegen."""
    dbf = load_database()
    mod = Model(dbf, ['C'], 'DIAMOND_A4')
    mod_vars = [v.N, v.P, v.T] + mod.site_fractions

    # Test builds functions only, since functions takes about 1 second to run.
    # Both lambda and llvm backends take a few seconds to build the derivatives
    # and are probably unnecessary to test.
    # XXX: SymEngine does not produce a zoo for this case
    assert zoo in list(mod.GM.atoms())
    build_functions(mod.GM, mod_vars, include_obj=True, include_grad=False, include_hess=False)

    int_cons = mod.get_internal_constraints()
    build_constraint_functions(mod_vars, int_cons)
