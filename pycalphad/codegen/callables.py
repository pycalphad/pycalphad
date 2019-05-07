import pycalphad.variables as v
from pycalphad.codegen.sympydiff_utils import build_functions
from pycalphad.core.utils import get_pure_elements, unpack_components, extract_parameters, get_state_variables
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.constraints import build_constraints
from itertools import repeat


def build_callables(dbf, comps, phases, models, conds, parameters=None, callables=None,
                    output='GM', build_gradients=True, build_hessians=False,
                    additional_statevars=None, verbose=False):
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
    models : dict
        Dictionary of {phase_name: Model subclass}
    parameters : dict, optional
        Maps SymPy Symbol to numbers, for overriding the values of parameters in the Database.
    callables : dict, optional
        Pre-computed callables
    output : str
        Output property of the particular Model to sample
    build_gradients : bool
        Whether or not to build gradient functions. Defaults to True.
    build_hessians : bool
        Whether or not to build Hessian functions. Defaults to False.
    additional_statevars : set or None
        State variables to include in the callables that may not be in the models

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
    additional_statevars = additional_statevars if additional_statevars is not None else set()
    parameters = parameters if parameters is not None else {}
    param_symbols, param_values = extract_parameters(parameters)
    comps = sorted(unpack_components(dbf, comps))
    pure_elements = get_pure_elements(dbf, comps)

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

    phase_records = {}

    state_variables = get_state_variables(models=models)
    state_variables |= additional_statevars
    state_variables = sorted(state_variables, key=str)

    for name in phases:
        mod = models[name]
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

        cfuncs = build_constraints(mod, state_variables + site_fracs, conds, parameters=param_symbols)
        _callables['internal_cons'][name] = cfuncs.internal_cons
        _callables['internal_jac'][name] = cfuncs.internal_jac
        _callables['internal_cons_hess'][name] = cfuncs.internal_cons_hess
        _callables['mp_cons'][name] = cfuncs.multiphase_cons
        _callables['mp_jac'][name] = cfuncs.multiphase_jac
        num_internal_cons = cfuncs.num_internal_cons
        num_multiphase_cons = cfuncs.num_multiphase_cons

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
    return phase_records, state_variables
