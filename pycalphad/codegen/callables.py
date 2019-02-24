import pycalphad.variables as v
from pycalphad import Model
from pycalphad.codegen.sympydiff_utils import build_functions
from pycalphad.core.utils import get_pure_elements, unpack_components, unpack_kwarg
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.constraints import build_constraints
from sympy import Symbol
import numpy as np
import operator
from itertools import repeat


def wrap_symbol(obj):
    if isinstance(obj, Symbol):
        return obj
    else:
        return Symbol(obj)


def build_callables(dbf, comps, phases, conds=None, model=None, parameters=None, callables=None,
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
    conds : dict or None
        Conditions for calculation
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
    conds = conds if conds is not None else {}
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
    # TODO: Conditions needing Hessians should probably have a 'second-order' tag or something
    build_hessians = any(str(cond).startswith('MU') for cond in conds.keys())

    callables = callables if callables is not None else {}
    _callables = {
        'massfuncs': {},
        'massgradfuncs': {},
        'masshessfuncs': {},
        'callables': {},
        'grad_callables': {},
        'hess_callables': {},
        'internal_cons': {},
        'internal_jac': {},
        'internal_cons_hess': {},
        'mp_cons': {},
        'mp_jac': {},
    }

    models = unpack_kwarg(model, default_arg=Model)
    param_symbols = [wrap_symbol(sym) for sym in param_symbols]
    phase_records = {}

    state_variables = set()
    for name in phases:
        mod = models[name]
        if isinstance(mod, type):
            models[name] = mod = mod(dbf, comps, name, parameters=param_symbols)
        state_variables |= set(mod.state_variables)

    unspecified_statevars = state_variables - set(conds.keys())
    if len(unspecified_statevars) > 0:
        raise ValueError('The following state variables must be specified: {0}'.format(unspecified_statevars))
        # TODO: T,P as free variables

    unused_statevars = set()
    for x in conds.keys():
        if (getattr(v, str(x), None) is not None) and not isinstance(x, v.ChemicalPotential):
            unused_statevars |= {x}
    unused_statevars -= state_variables
    if len(unused_statevars) > 0:
        state_variables |= unused_statevars

    state_variables = sorted(state_variables, key=str)
    _callables['state_variables'] = state_variables

    for name in phases:
        mod = models[name]
        if isinstance(mod, type):
            models[name] = mod = mod(dbf, comps, name, parameters=param_symbols)
        site_fracs = mod.site_fractions
        try:
            out = getattr(mod, output)
        except AttributeError:
            raise AttributeError('Missing Model attribute {0} specified for {1}'
                                 .format(output, mod.__class__))

        if callables.get('callables', {}).get(name, False) and \
                ((not build_gradients) or callables.get('grad_callables', {}).get(name, False)) and \
                ((not build_hessians) or callables.get('hess_callables', {}).get(name, False)):
            _callables['callables'][name] = callables['callables'][name]
            _callables['grad_callables'][name] = callables['grad_callables'].get(name, None)
            _callables['hess_callables'][name] = callables['hess_callables'].get(name, None)
        else:
            # Build the callables of the output
            # Only force undefineds to zero if we're not overriding them
            undefs = {x for x in out.free_symbols if not isinstance(x, v.StateVariable)} - set(param_symbols)
            undef_vals = repeat(0., len(undefs))
            out = out.xreplace(dict(zip(undefs, undef_vals)))
            build_output = build_functions(out, tuple(state_variables + site_fracs), parameters=param_symbols,
                                           include_grad=build_gradients, include_hess=build_hessians)
            cf, gf, hf = build_output.func, build_output.grad, build_output.hess
            _callables['callables'][name] = cf
            _callables['grad_callables'][name] = gf
            _callables['hess_callables'][name] = hf

        if callables.get('massfuncs', {}).get(name, False) and \
                ((not build_gradients) or callables.get('massgradfuncs', {}).get(name, False)) and \
                ((not build_hessians) or callables.get('masshessfuncs', {}).get(name, False)):
            _callables['massfuncs'][name] = callables['massfuncs'][name]
            _callables['massgradfuncs'][name] = callables['massgradfuncs'].get(name, None)
            _callables['masshessfuncs'][name] = callables['masshessfuncs'].get(name, None)
        else:
            # Build the callables for mass
            # TODO: In principle, we should also check for undefs in mod.moles()
            mcf, mgf, mhf = zip(*[build_functions(mod.moles(el), state_variables + site_fracs,
                                                  include_obj=True,
                                                  include_grad=build_gradients,
                                                  include_hess=build_hessians,
                                                  parameters=param_symbols)
                                  for el in pure_elements])
            if all(x is None for x in mgf):
                mgf = None
            if all(x is None for x in mhf):
                mhf = None
            _callables['massfuncs'][name] = mcf
            _callables['massgradfuncs'][name] = mgf
            _callables['masshessfuncs'][name] = mhf
        if not callables.get('phase_records', {}).get(name, False):
            pv = param_values
        else:
            # Copy parameter values from old PhaseRecord, if it exists
            pv = callables['phase_records'][name].parameters

        if len(conds) > 0:
            cfuncs = build_constraints(mod, state_variables + site_fracs, conds, parameters=param_symbols)
            _callables['internal_cons'][name] = cfuncs.internal_cons
            _callables['internal_jac'][name] = cfuncs.internal_jac
            _callables['internal_cons_hess'][name] = cfuncs.internal_cons_hess
            _callables['mp_cons'][name] = cfuncs.multiphase_cons
            _callables['mp_jac'][name] = cfuncs.multiphase_jac
            num_internal_cons = cfuncs.num_internal_cons
            num_multiphase_cons = cfuncs.num_multiphase_cons
        else:
            _callables['internal_cons'][name] = None
            _callables['internal_jac'][name] = None
            _callables['internal_cons_hess'][name] = None
            _callables['mp_cons'][name] = None
            _callables['mp_jac'][name] = None
            num_internal_cons = 0
            num_multiphase_cons = 0
        phase_records[name.upper()] = PhaseRecord(comps, state_variables, site_fracs, pv,
                                                  _callables['callables'][name],
                                                  _callables['grad_callables'][name],
                                                  _callables['hess_callables'][name],
                                                  _callables['massfuncs'][name],
                                                  _callables['massgradfuncs'][name],
                                                  _callables['masshessfuncs'][name],
                                                  _callables['internal_cons'][name],
                                                  _callables['internal_jac'][name],
                                                  _callables['internal_cons_hess'][name],
                                                  _callables['mp_cons'][name],
                                                  _callables['mp_jac'][name],
                                                  num_internal_cons,
                                                  num_multiphase_cons)
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
