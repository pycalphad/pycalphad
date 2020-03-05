"""
The utils test module contains tests for pycalphad utilities.
"""

import pytest
from symengine.lib.symengine_wrapper import LambdaDouble, LLVMDouble
from pycalphad import Database, Model
from pycalphad.core.utils import filter_phases, unpack_components, instantiate_models
from pycalphad.codegen.sympydiff_utils import build_functions, build_constraint_functions
from pycalphad.tests.datasets import ALNIPT_TDB, ALCRNI_TDB

ALNIPT_DBF = Database(ALNIPT_TDB)
ALCRNI_DBF = Database(ALCRNI_TDB)

def test_filter_phases_removes_disordered_phases_from_order_disorder():
    """Databases with order-disorder models should have the disordered phases be filtered if candidate_phases kwarg is not passed to filter_phases.
    If candidate_phases kwarg is passed, disordered phases just are filtered if respective ordered phases are inactive"""
    all_phases = set(ALNIPT_DBF.phases.keys())
    filtered_phases = set(filter_phases(ALNIPT_DBF, unpack_components(ALNIPT_DBF, ['AL', 'NI', 'PT', 'VA'])))
    assert all_phases.difference(filtered_phases) == {'FCC_A1'}
    comps = unpack_components(ALCRNI_DBF, ['NI', 'AL', 'CR', 'VA'])
    filtered_phases = set(filter_phases(ALCRNI_DBF, comps, ['FCC_A1', 'L12_FCC', 'LIQUID', 'BCC_A2']))
    assert filtered_phases == {'L12_FCC', 'LIQUID', 'BCC_A2'}
    filtered_phases = set(filter_phases(ALCRNI_DBF, comps, ['FCC_A1', 'LIQUID', 'BCC_A2']))
    assert filtered_phases == {'FCC_A1', 'LIQUID', 'BCC_A2'}
    filtered_phases = set(filter_phases(ALCRNI_DBF, comps, ['FCC_A1']))
    assert filtered_phases == {'FCC_A1'}


def test_filter_phases_removes_phases_with_inactive_sublattices():
    """Phases that have no active components in any sublattice should be filtered"""
    all_phases = set(ALNIPT_DBF.phases.keys())
    filtered_phases = set(filter_phases(ALNIPT_DBF, unpack_components(ALNIPT_DBF, ['AL', 'NI', 'VA'])))
    assert all_phases.difference(filtered_phases) == {'FCC_A1', 'PT8AL21', 'PT5AL21', 'PT2AL', 'PT2AL3', 'PT5AL3', 'ALPT2'}


def test_instantiate_models_only_returns_desired_phases():
    """instantiate_models should only return phases passed"""
    comps = ['AL', 'NI', 'VA']
    phases = ['FCC_A1', 'LIQUID']

    # models are overspecified w.r.t. phases
    too_many_phases = ['FCC_A1', 'LIQUID', 'AL3NI1']
    too_many_models = {phase: Model(ALNIPT_DBF, comps, phase) for phase in too_many_phases}
    inst_mods = instantiate_models(ALNIPT_DBF, comps, phases, model=too_many_models)
    assert len(inst_mods) == len(phases)

    # models are underspecified w.r.t. phases
    too_few_phases = ['FCC_A1']
    too_few_models = {phase: Model(ALNIPT_DBF, comps, phase) for phase in too_few_phases}
    inst_mods = instantiate_models(ALNIPT_DBF, comps, phases, model=too_few_models)
    assert len(inst_mods) == len(phases)


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
