"""
This module contains routines for fitting new CALPHAD models.
"""

import numpy as np
import xarray
from pycalphad import calculate, equilibrium, Model, Database
from pycalphad.core.utils import make_callable, generate_dof
import pycalphad.variables as v
from sympy import Symbol
import matplotlib.pyplot as plt
import itertools
import functools
from collections import OrderedDict, namedtuple
import json


def setup_dataset(file_obj, dbf, params, mode=None):
    # params should be a list of pymc variables corresponding to parameters
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
                                     itertools.chain(param_vars, indep_vars, site_fracs[name]), mode=mode)
                 for name, mod in fit_models.items()}
    extra_conds = OrderedDict({key: np.atleast_1d(value) for key, value in data['conditions'].items()})
    exp_values = xarray.DataArray(np.array(data['values'], dtype=np.float),
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

    return compute_error, compute_values, exp_values, data

Dataset = namedtuple('Dataset', ['error_func', 'calc_func', 'exp_data', 'json'])


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
    (pymc.Model, dict of Dataset)
    """
    # TODO: Figure out a better solution than hiding an import here
    import pymc
    params = sorted(params, key=str)
    dataset_error_funcs = []
    dataset_est_variances = []
    datasets = []
    for idx, fname in enumerate(dataset_names):
        with open(fname) as file_:
            datasets.append(Dataset(*setup_dataset(file_, dbf, params)))
            dataset_est_variances.append(np.var(datasets[-1].exp_data.values))
            dataset_error_funcs.append(datasets[-1].error_func)
    # For 1/var**2, choose gamma(alpha=10*est_var, beta=10)
    #dataset_variance = pymc.Gamma('dataset_variance',
    #                              alpha=np.asarray(dataset_est_variances, dtype=np.float),
    #                              beta=np.asarray(dataset_est_variances, dtype=np.float),
    #                              size=len(dataset_names))
    function_namespace = {'zeros_like': np.zeros_like, 'square': np.square, 'divide': np.divide,
                          'dataset_est_variances': dataset_est_variances, 'dataset_names': dataset_names}
    # TODO: Is there a security issue with passing the output of str(x) to exec?
    function_namespace.update([(str(param), param) for param in params])
    param_kwarg_names = ','.join([str(param) + '=' + str(param) for param in params])
    param_arg_names = ','.join([str(param) for param in params])

    function_namespace.update({'dataset_error_funcs': dataset_error_funcs})
    # Now we have to do some metaprogramming to get the variable names to bind properly
    # This code doesn't yet allow custom distributions for the error
    error_func_code = """def error({0}):
    result = zeros_like(dataset_names, dtype='float')
    for idx in range(len(dataset_names)):
        result[idx] = divide(square(dataset_error_funcs[idx]({1})).mean(), dataset_est_variances[idx])
    return -result.sum()""".format(param_kwarg_names, param_arg_names)
    error_func_code = compile(error_func_code, '<string>', 'exec')
    exec(error_func_code, function_namespace)
    error = pymc.potential(function_namespace['error'])
    mod = pymc.Model([function_namespace[str(param)] for param in params] + [error])
    return mod, datasets


def _map_internal_dof(input_database, components, phase_name, points):
    """
    Map matrix of internal degrees of freedom to global compositions.
    """
    # Map the internal degrees of freedom to global coordinates
    # Normalize site ratios by the sum of site ratios times a factor
    # related to the site fraction of vacancies
    phase_obj = input_database.phases[phase_name]
    site_ratio_normalization = np.zeros(points.shape[:-1])
    phase_compositions = np.empty(points.shape[:-1] + (len(components),))
    variables, sublattice_dof = generate_dof(phase_obj, components)
    for idx, sublattice in enumerate(phase_obj.constituents):
        vacancy_column = np.ones(points.shape[:-1])
        if 'VA' in set(sublattice):
            var_idx = variables.index(v.SiteFraction(phase_obj.name, idx, 'VA'))
            vacancy_column -= points[..., :, var_idx]
        site_ratio_normalization += phase_obj.sublattices[idx] * vacancy_column

    for col, comp in enumerate(components):
        avector = [float(vxx.species == comp) *
                   phase_obj.sublattices[vxx.sublattice_index] for vxx in variables]
        phase_compositions[..., :, col] = np.divide(np.dot(points[..., :, :], avector),
                                                    site_ratio_normalization)
    return phase_compositions


def plot_results(input_database, datasets, params, databases=None):
    """
    Generate figures using the datasets and trace of the parameters.
    A dict of label->Database objects may be provided as a kwarg.
    """
    # Add extra broadcast dimensions for T, P, and 'points'
    param_tr = [i[None, None, None].T for i in params.values()]

    def plot_key(obj):
        plot = obj.json.get('plot', None)
        return (plot['x'], plot['y']) if plot else None
    datasets = sorted(datasets, key=plot_key)
    databases = dict() if databases is None else databases
    for plot_data_type, data_group in itertools.groupby(datasets, key=plot_key):
        if plot_data_type is None:
            continue
        figure = plt.figure(figsize=(15, 12))
        data_group = list(data_group)
        x, y = plot_data_type
        # All of data_group should be calculating the same thing...
        # Don't show fits below 300 K since they're currently meaningless
        # TODO: Calls to flatten() should actually be slicing operations
        # We can get away with it for now since all datasets will be 2D
        fit = data_group[0].calc_func(*param_tr).sel(T=slice(300, None))
        mu = fit[y].values.mean(axis=0).flatten()
        sigma = 2 * fit[y].values.std(axis=0).flatten()
        figure.gca().plot(fit[x].values.flatten(), mu, '-k', label='This work')
        figure.gca().fill_between(fit[x].values.flatten(), mu - sigma, mu + sigma, color='lightgray')
        for data in data_group:
            plot_label = data.json['plot'].get('name', None)
            figure.gca().plot(data.exp_data[x].values, data.exp_data.values.flatten(), label=plot_label)
        for label, dbf in databases.items():
            # TODO: Relax this restriction
            if data_group[0].json['solver']['mode'] != 'manual':
                continue
            conds = data_group[0].json['conditions']
            conds['T'] = np.array(conds['T'])
            conds['T'] = conds['T'][conds['T'] >= 300.]
            for key in conds.keys():
                if key not in ['T', 'P']:
                    raise ValueError('Invalid conditions in JSON file')
            # To work around differences in sublattice models, relax the internal dof
            global_comps = sorted(set(data_group[0].json['components']) - set(['VA']))
            compositions = \
                _map_internal_dof(input_database,
                                  sorted(data_group[0].json['components']),
                                  data_group[0].json['phases'][0],
                                  np.atleast_2d(
                                      data_group[0].json['solver']['sublattice_configuration']).astype(np.float))
            # Tiny perturbation to work around a bug in lower_convex_hull (gh-28)
            compare_conds = {v.X(comp): np.add(compositions[:, idx], 1e-4).flatten().tolist()
                             for idx, comp in enumerate(global_comps[:-1])}
            compare_conds.update({v.__dict__[key]: value for key, value in conds.items()})
            # We only want to relax the internal dof at the lowest temperature
            # This will help us capture the most related sublattice config since solver mode=manual
            # probably means this is first-principles data
            compare_conds[v.T] = 300.
            eqres = equilibrium(dbf, data_group[0].json['components'],
                                str(data_group[0].json['phases'][0]), compare_conds, verbose=False)
            internal_dof = sum(map(len, dbf.phases[data_group[0].json['phases'][0]].constituents))
            largest_phase_fraction = eqres['NP'].values.argmax()
            eqpoints = eqres['Y'].values[..., largest_phase_fraction, :internal_dof]
            result = calculate(dbf, data_group[0].json['components'],
                               str(data_group[0].json['phases'][0]), output=y,
                               points=eqpoints, **conds)
            # Don't show CALPHAD results below 300 K because they're meaningless right now
            result = result.sel(T=slice(300, None))
            figure.gca().plot(result[x].values.flatten(), result[y].values.flatten(), label=label)
        label_mapping = dict(x=x, y=y)
        label_mapping['CPM'] = 'Molar Heat Capacity (J/mol-atom-K)'
        label_mapping['SM'] = 'Molar Entropy (J/mol-atom-K)'
        label_mapping['HM'] = 'Molar Enthalpy (J/mol-atom)'
        label_mapping['T'] = 'Temperature (K)'
        label_mapping['P'] = 'Pressure (Pa)'
        figure.gca().set_xlabel(label_mapping[x], fontsize=20)
        figure.gca().set_ylabel(label_mapping[y], fontsize=20)
        figure.gca().tick_params(axis='both', which='major', labelsize=20)
        figure.gca().legend(loc='best', fontsize=16)
        figure.canvas.draw()
        yield figure
    plt.show()
