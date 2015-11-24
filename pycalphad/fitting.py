"""
This module contains routines for fitting new CALPHAD models.
"""

import numpy as np
import xray
from pycalphad import calculate, Model
from pycalphad.core.utils import make_callable
import pycalphad.variables as v
from sympy import Symbol
import itertools
import functools
from collections import OrderedDict
import json


def setup_dataset(file_obj, dbf, params):
    # params should be a list of pymc3 variables corresponding to parameters
    data = json.load(file_obj)
    if data['solver']['mode'] != 'manual':
        raise NotImplemented
    fit_models = {name: Model(dbf, data['components'], name) for name in data['phases']}
    param_vars = []
    for key, mod in fit_models.items():
        param_vars.extend(sorted(set(mod.ast.atoms(Symbol)) - set(mod.variables), key=str))
    param_vars = sorted(param_vars, key=str)
    if len(params) != len(param_vars):
        raise ValueError('Input parameter vector length doesn\'t match the free parameters'
                         ' in the phase models: {0} != {1}'.format(len(params), len(param_vars)))
    indep_vars = [v.P, v.T]
    site_fracs = {key: sorted(mod.ast.atoms(v.SiteFraction), key=str) for key, mod in fit_models.items()}
    # Call this from a routine that pulls in all datasets and generates the variance vars + Potentials
    callables = {name: make_callable(getattr(mod, data['output']),
                                     itertools.chain(param_vars, indep_vars, site_fracs[name]))
                 for name, mod in fit_models.items()}
    extra_conds = OrderedDict({key: np.atleast_1d(value) for key, value in data['conditions'].items()})
    exp_values = xray.DataArray(np.array(data['values'], dtype=np.float),
                                dims=list(extra_conds.keys())+['points'], coords=extra_conds)

    def compute_error(*args):
        prefill_callables = {key: functools.partial(*itertools.chain([func], args[:len(params)]))
                             for key, func in callables.items()}
        result = calculate(dbf, data['components'], data['phases'], output=data['output'],
                           points=np.atleast_2d(data['solver']['sublattice_configuration']).astype(np.float),
                           callables=prefill_callables, model=fit_models, **extra_conds)
        # Eliminate data below 300 K for now
        error = (result[data['output']] - exp_values).sel(T=slice(300, None)).values.flatten()
        return error

    def compute_values(*args):
        prefill_callables = {key: functools.partial(*itertools.chain([func], args[:len(params)]))
                             for key, func in callables.items()}
        result = calculate(dbf, data['components'], data['phases'], output=data['output'],
                           points=np.atleast_2d(data['solver']['sublattice_configuration']).astype(np.float),
                           callables=prefill_callables, model=fit_models, **extra_conds)
        return result

    return compute_error, compute_values, exp_values


def build_pymc_model(dbf, dataset_names, params):
    """
    Build a pymc (2.x) Model using the specified data and parameters.

    Parameters
    ----------
    dbf : Database
        Database with defined parameters, except for params.
    dataset_names : list of str
        List of paths to pycalphad JSON files.
    params : list of pymc.Stochastic
        Parameters to fit.

    Returns
    -------
    pymc.Model
    """
    # TODO: Figure out a better solution than hiding an import here
    import pymc
    params = sorted(params, key=str)
    # Should users be able to specify their own variance parameter?
    # Ideally this prior would come from the dataset itself
    dataset_variance = pymc.Gamma('dataset_variance',
                                  alpha=np.full_like(dataset_names, 0.1, dtype=np.float),
                                  beta=np.full_like(dataset_names, 0.1, dtype=np.float),
                                  size=len(dataset_names))
    dataset_error_funcs = []
    function_namespace = {'zeros_like': np.zeros_like, 'square': np.square, 'divide': np.divide,
                          'dataset_variance': dataset_variance, 'dataset_names': dataset_names}
    # TODO: Is there a security issue with passing the output of str(x) to exec?
    function_namespace.update([(str(param), param) for param in params])
    param_kwarg_names = ','.join([str(param) + '=' + str(param) for param in params+[dataset_variance]])
    param_arg_names = ','.join([str(param) for param in params])
    for idx, fname in enumerate(dataset_names):
        with open(fname) as file_:
            error_func, calc_func, exp_data = setup_dataset(file_, dbf, params)
            dataset_error_funcs.append(error_func)
    function_namespace.update({'dataset_error_funcs': dataset_error_funcs})
    # Now we have to do some metaprogramming to get the variable names to bind properly
    # This code doesn't yet allow custom distributions for the error
    error_func_code = """def error({0}):
    result = zeros_like(dataset_names, dtype='float')
    for idx in range(len(dataset_names)):
        result[idx] = divide(square(dataset_error_funcs[idx]({1})).mean(), dataset_variance[idx])
    return -result""".format(param_kwarg_names, param_arg_names)
    error_func_code = compile(error_func_code, '<string>', 'exec')
    exec(error_func_code, function_namespace)
    error = pymc.potential(function_namespace['error'])
    mod = pymc.Model([function_namespace[str(param)] for param in params] + [function_namespace['dataset_variance'],
                     error])
    return mod
