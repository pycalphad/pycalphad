import pycalphad.variables as v
from pycalphad.codegen.sympydiff_utils import build_functions
from pycalphad.core.utils import get_pure_elements, unpack_components, \
    extract_parameters, get_state_variables, wrap_symbol
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.constraints import build_constraints
from itertools import repeat
import warnings


def build_callables(dbf, comps, phases, models, parameter_symbols=None,
                    output='GM', build_gradients=True, build_hessians=False,
                    additional_statevars=None):
    """
    Create a compiled callables dictionary.

    Parameters
    ----------
    dbf : Database
        A Database object
    comps : list
        List of component names
    phases : list
        List of phase names
    models : dict
        Dictionary of {phase_name: Model subclass}
    parameter_symbols : list, optional
        List of string or SymEngine Symbols that will be overridden in the callables.
    output : str, optional
        Output property of the particular Model to sample. Defaults to 'GM'
    build_gradients : bool, optional
        Whether or not to build gradient functions. Defaults to True.
    build_hessians : bool, optional
        Whether or not to build Hessian functions. Defaults to False.
    additional_statevars : set, optional
        State variables to include in the callables that may not be in the models (e.g. from conditions)
    verbose : bool, optional
        Print the name of the phase when its callables are built

    Returns
    -------
    callables : dict
        Dictionary of keyword argument callables to pass to equilibrium.
        Maps {'output' -> {'function' -> {'phase_name' -> AutowrapFunction()}}.

    Notes
    -----
    *All* the state variables used in calculations must be specified.
    If these are not specified as state variables of the models (e.g. often the
    case for v.N), then it must be supplied by the additional_statevars keyword
    argument.

    Examples
    --------
    >>> from pycalphad import Database, equilibrium, variables as v
    >>> from pycalphad.codegen.callables import build_callables
    >>> from pycalphad.core.utils import instantiate_models
    >>> dbf = Database('AL-NI.tdb')
    >>> comps = ['AL', 'NI', 'VA']
    >>> phases = ['LIQUID', 'AL3NI5', 'AL3NI2', 'AL3NI']
    >>> models = instantiate_models(dbf, comps, phases)
    >>> callables = build_callables(dbf, comps, phases, models, additional_statevars={v.P, v.T, v.N})
    >>> 'GM' in callables.keys()
    True
    >>> 'massfuncs' in callables['GM']
    True
    >>> conditions = {v.P: 101325, v.T: 2500, v.X('AL'): 0.2}
    >>> equilibrium(dbf, comps, phases, conditions, callables=callables)
    """
    additional_statevars = set(additional_statevars) if additional_statevars is not None else set()
    parameter_symbols = parameter_symbols if parameter_symbols is not None else []
    parameter_symbols = sorted([wrap_symbol(x) for x in parameter_symbols], key=str)
    comps = sorted(unpack_components(dbf, comps))
    pure_elements = get_pure_elements(dbf, comps)

    _callables = {
        'massfuncs': {},
        'massgradfuncs': {},
        'masshessfuncs': {},
        'formulamolefuncs': {},
        'formulamolegradfuncs': {},
        'formulamolehessfuncs': {},
        'callables': {},
        'grad_callables': {},
        'hess_callables': {},
        'internal_cons_func': {},
        'internal_cons_jac': {},
        'internal_cons_hess': {},
    }

    state_variables = get_state_variables(models=models)
    state_variables |= additional_statevars
    if not {v.T, v.P, v.N}.issubset(state_variables):
        warnings.warn("State variables in `build_callables` are not {{N, P, T}}, but {}. This can lead to incorrectly "
                      "calculated values if the state variables used to call the generated functions do not match the "
                      "state variables used to create them. State variables can be added with the "
                      "`additional_statevars` argument.".format(state_variables))
    state_variables = sorted(state_variables, key=str)

    for name in phases:
        mod = models[name]
        site_fracs = mod.site_fractions
        try:
            out = getattr(mod, output)
        except AttributeError:
            raise AttributeError('Missing Model attribute {0} specified for {1}'
                                 .format(output, mod.__class__))

        # Build the callables of the output
        # Only force undefineds to zero if we're not overriding them
        undefs = {x for x in out.free_symbols if not isinstance(x, v.StateVariable)} - set(parameter_symbols)
        undef_vals = repeat(0., len(undefs))
        out = out.xreplace(dict(zip(undefs, undef_vals)))
        build_output = build_functions(out, tuple(state_variables + site_fracs), parameters=parameter_symbols,
                                       include_grad=build_gradients, include_hess=build_hessians)
        cf, gf, hf = build_output.func, build_output.grad, build_output.hess
        _callables['callables'][name] = cf
        _callables['grad_callables'][name] = gf
        _callables['hess_callables'][name] = hf

        # Build the callables for mass
        # TODO: In principle, we should also check for undefs in mod.moles()
        mcf, mgf, mhf = zip(*[build_functions(mod.moles(el), state_variables + site_fracs,
                                              include_obj=True,
                                              include_grad=build_gradients,
                                              include_hess=build_hessians,
                                              parameters=parameter_symbols)
                              for el in pure_elements])

        _callables['massfuncs'][name] = mcf
        _callables['massgradfuncs'][name] = mgf
        _callables['masshessfuncs'][name] = mhf

        # Build the callables for moles per formula unit
        # TODO: In principle, we should also check for undefs in mod.moles()
        fmcf, fmgf, fmhf = zip(*[build_functions(mod.moles(el, per_formula_unit=True), state_variables + site_fracs,
                                                 include_obj=True,
                                                 include_grad=build_gradients,
                                                 include_hess=build_hessians,
                                                 parameters=parameter_symbols)
                                 for el in pure_elements])

        _callables['formulamolefuncs'][name] = fmcf
        _callables['formulamolegradfuncs'][name] = fmgf
        _callables['formulamolehessfuncs'][name] = fmhf
    return {output: _callables}


def build_phase_records(dbf, comps, phases, state_variables, models, output='GM',
                        callables=None, parameters=None, verbose=False,
                        build_gradients=True, build_hessians=True
                        ):
    """
    Combine compiled callables and callables from conditions into PhaseRecords.

    Parameters
    ----------
    dbf : Database
        A Database object
    comps : List[Union[str, v.Species]]
        List of active pure elements or species.
    phases : list
        List of phase names
    state_variables : Iterable[v.StateVariable]
        State variables used to produce the generated functions.
    models : Mapping[str, Model]
        Mapping of phase names to model instances
    parameters : dict, optional
        Maps SymEngine Symbol to numbers, for overriding the values of parameters in the Database.
    callables : dict, optional
        Pre-computed callables. If None are passed, they will be built.
        Maps {'output' -> {'function' -> {'phase_name' -> AutowrapFunction()}}
    output : str
        Output property of the particular Model to sample
    verbose : bool, optional
        Print the name of the phase when its callables are built
    build_gradients : bool
        Whether or not to build gradient functions. Defaults to False. Only
        takes effect if callables are not passed.
    build_hessians : bool
        Whether or not to build Hessian functions. Defaults to False. Only
        takes effect if callables are not passed.

    Returns
    -------
    dict
        Dictionary mapping phase names to PhaseRecord instances.

    Notes
    -----
    If callables are passed, don't rebuild them. This means that the callables
    are not checked for incompatibility. Users of build_callables are
    responsible for ensuring that the state variables, parameters and models
    used to construct the callables are compatible with the ones used to
    build the constraints and phase records.

    """
    comps = sorted(unpack_components(dbf, comps))
    parameters = parameters if parameters is not None else {}
    callables = callables if callables is not None else {}
    _constraints = {
        'internal_cons_func': {},
        'internal_cons_jac': {},
        'internal_cons_hess': {},
    }
    phase_records = {}
    state_variables = sorted(get_state_variables(models=models, conds=state_variables), key=str)
    param_symbols, param_values = extract_parameters(parameters)

    if callables.get(output) is None:
        callables = build_callables(dbf, comps, phases, models,
                                    parameter_symbols=parameters.keys(), output=output,
                                    additional_statevars=state_variables,
                                    build_gradients=False,
                                    build_hessians=False)
    # Temporary solution. PhaseRecord needs rework: https://github.com/pycalphad/pycalphad/pull/329#discussion_r634579356
    formulacallables = build_callables(dbf, comps, phases, models,
                                       parameter_symbols=parameters.keys(), output='G',
                                       additional_statevars=state_variables,
                                       build_gradients=build_gradients,
                                       build_hessians=build_hessians)

    # If a vector of parameters is specified, only pass the first row to the PhaseRecord
    # Future callers of PhaseRecord.obj_parameters_2d() can pass the full param_values array as an argument
    if len(param_values.shape) > 1:
        param_values = param_values[0]

    for name in phases:
        mod = models[name]
        site_fracs = mod.site_fractions
        # build constraint functions
        cfuncs = build_constraints(mod, state_variables + site_fracs, parameters=param_symbols)
        _constraints['internal_cons_func'][name] = cfuncs.internal_cons_func
        _constraints['internal_cons_jac'][name] = cfuncs.internal_cons_jac
        _constraints['internal_cons_hess'][name] = cfuncs.internal_cons_hess
        num_internal_cons = cfuncs.num_internal_cons

        phase_records[name.upper()] = PhaseRecord(comps, state_variables, site_fracs, param_values,
                                                  callables[output]['callables'][name],
                                                  formulacallables['G']['callables'][name],
                                                  formulacallables['G']['grad_callables'][name],
                                                  formulacallables['G']['hess_callables'][name],
                                                  callables[output]['massfuncs'][name],
                                                  formulacallables['G']['formulamolefuncs'][name],
                                                  formulacallables['G']['formulamolegradfuncs'][name],
                                                  formulacallables['G']['formulamolehessfuncs'][name],
                                                  _constraints['internal_cons_func'][name],
                                                  _constraints['internal_cons_jac'][name],
                                                  _constraints['internal_cons_hess'][name],
                                                  num_internal_cons)

        if verbose:
            print(name + ' ')
    return phase_records
