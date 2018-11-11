import pycalphad.variables as v
from pycalphad import Model
from pycalphad.codegen.sympydiff_utils import build_functions
from pycalphad.core.utils import get_pure_elements, unpack_components, unpack_kwarg
from pycalphad.core.phase_rec import PhaseRecord_from_cython
from sympy import Symbol
import numpy as np
import operator
from itertools import repeat
from collections import defaultdict


def wrap_symbol(obj):
    if isinstance(obj, Symbol):
        return obj
    else:
        return Symbol(obj)

def build_callables(dbf, comps, phases, model=None, parameters=None, callables=None,
                    output='GM', build_gradients=True, verbose=False):
    """
    Create dictionaries of callable dictionaries and PhaseRecords.

    Parameters
    ----------
    dbf : Database
        A Database object
    comps : list
        List of component names
    phases : list
        List of phase names
    model : dict or type
        Dictionary of {phase_name: Model subclass} or a type corresponding to a
        Model subclass. Defaults to ``Model``.
    parameters : dict, optional
        Maps SymPy Symbol to numbers, for overriding the values of parameters in the Database.
    callables : dict, optional
        Pre-computed callables
    output : str
        Output property of the particular Model to sample
    build_gradients : bool
        Whether or not to build gradient functions. Defaults to True.

    verbose : bool
        Print the name of the phase when its callables are built

    Returns
    -------
    callables : dict
        Dictionary of keyword argument callables to pass to equilibrium.

    Example
    -------
    >>> dbf = Database('AL-NI.tdb')
    >>> comps = ['AL', 'NI', 'VA']
    >>> phases = ['FCC_L12', 'BCC_B2', 'LIQUID', 'AL3NI5', 'AL3NI2', 'AL3NI']
    >>> callables = build_callables(dbf, comps, phases)
    >>> equilibrium(dbf, comps, phases, conditions, **callables)
    """
    parameters = parameters if parameters is not None else {}
    if len(parameters) > 0:
        param_symbols, param_values = zip(*[(key, val) for key, val in sorted(parameters.items(),
                                                                              key=operator.itemgetter(0))])
        param_values = np.asarray(param_values, dtype=np.float64)
    else:
        param_symbols = []
        param_values = np.empty(0)
    comps = sorted(unpack_components(dbf, comps))
    pure_elements = get_pure_elements(dbf, comps)

    callables = callables if callables is not None else {}
    _callables = {
        'massfuncs': {},
        'massgradfuncs': {},
        'callables': {},
        'grad_callables': {},
        'hess_callables': defaultdict(lambda: None),
    }

    models = unpack_kwarg(model, default_arg=Model)
    param_symbols = [wrap_symbol(sym) for sym in param_symbols]
    phase_records = {}
    # create models
    for name in phases:
        mod = models[name]
        if isinstance(mod, type):
            models[name] = mod = mod(dbf, comps, name, parameters=parameters)
        site_fracs = mod.site_fractions
        variables = sorted(site_fracs, key=str)
        try:
            out = getattr(mod, output)
        except AttributeError:
            raise AttributeError('Missing Model attribute {0} specified for {1}'
                                 .format(output, mod.__class__))

        if callables.get('callables', {}).get(name, False) and \
                ((not build_gradients) or callables.get('grad_callables',{}).get(name, False)):
            _callables['callables'][name] = callables['callables'][name]
            if build_gradients:
                _callables['grad_callables'][name] = callables['grad_callables'][name]
            else:
                _callables['grad_callables'][name] = None
        else:
            # Build the callables of the output
            # Only force undefineds to zero if we're not overriding them
            undefs = {x for x in out.free_symbols if not isinstance(x, v.StateVariable)} - set(param_symbols)
            undef_vals = repeat(0., len(undefs))
            out = out.xreplace(dict(zip(undefs, undef_vals)))
            build_output = build_functions(out, tuple([v.P, v.T] + site_fracs), parameters=param_symbols,
                                           include_grad=build_gradients)
            if build_gradients:
                cf, gf = build_output
            else:
                cf = build_output
                gf = None
            _callables['callables'][name] = cf
            _callables['grad_callables'][name] = gf

        if callables.get('massfuncs', {}).get(name, False) and \
                ((not build_gradients) or callables.get('massgradfuncs', {}).get(name, False)):
            _callables['massfuncs'][name] = callables['massfuncs'][name]
            if build_gradients:
                _callables['massgradfuncs'][name] = callables['massgradfuncs'][name]
            else:
                _callables['massgradfuncs'][name] = None
        else:
            # Build the callables for mass
            # TODO: In principle, we should also check for undefs in mod.moles()

            if build_gradients:
                mcf, mgf = zip(*[build_functions(mod.moles(el), [v.P, v.T] + variables,
                                                 include_obj=True,
                                                 include_grad=build_gradients,
                                                 parameters=param_symbols)
                                 for el in pure_elements])
            else:
                mcf = tuple([build_functions(mod.moles(el), [v.P, v.T] + variables,
                                             include_obj=True,
                                             include_grad=build_gradients,
                                             parameters=param_symbols)
                             for el in pure_elements])
                mgf = None
            _callables['massfuncs'][name] = mcf
            _callables['massgradfuncs'][name] = mgf
        if not callables.get('phase_records', {}).get(name, False):
            pv = param_values
        else:
            # Copy parameter values from old PhaseRecord, if it exists
            pv = callables['phase_records'][name].parameters
        phase_records[name.upper()] = PhaseRecord_from_cython(comps, variables,
                                                              np.array(dbf.phases[name].sublattices,
                                                                       dtype=np.float),
                                                              pv,
                                                              _callables['callables'][name],
                                                              _callables['grad_callables'][name],
                                                              _callables['hess_callables'][name],
                                                              _callables['massfuncs'][name],
                                                              _callables['massgradfuncs'][name])
        if verbose:
            print(name + ' ')

    # Update PhaseRecords with any user-specified parameter values, in case we skipped the build phase
    # We assume here that users know what they are doing, and pass compatible combinations of callables and parameters
    # See discussion in gh-192 for details
    if len(param_values) > 0:
        for prx_name in phase_records:
            if len(phase_records[prx_name].parameters) != len(param_values):
                raise ValueError('User-specified callables and parameters are incompatible')
            phase_records[prx_name].parameters = param_values
    # finally, add the models to the callables
    _callables['model'] = dict(models)
    _callables['phase_records'] = phase_records
    return _callables
