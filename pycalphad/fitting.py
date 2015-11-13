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
    exp_values = xray.DataArray(np.array(data['values'], dtype=np.float)[..., None],
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
