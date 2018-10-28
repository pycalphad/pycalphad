import pycalphad.variables as v
from pycalphad import Model
from pycalphad.codegen.sympydiff_utils import build_functions
from pycalphad.core.utils import get_pure_elements, unpack_components, unpack_kwarg
from pycalphad.core.phase_rec import PhaseRecord_from_cython
from sympy import Symbol
import numpy as np
import sys
PY2 = sys.version_info[0] == 2

if PY2:
    string_types = basestring
else:
    string_types = str


def build_callables(dbf, comps, phases, model=None, param_symbols=None, output='GM', build_gradients=True):
    """
    Create a dictionary of callable dictionaries for phases in equilibrium

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
    param_symbols : list
        SymPy Symbol objects that will be preserved in the callable functions.
    output : str
        Output property of the particular Model to sample
    build_gradients : bool
        Whether or not to build gradient functions. Defaults to True.

    Returns
    -------
    dict
        Dictionary of keyword argument callables to pass to equilibrium.

    Example
    -------
    >>> dbf = Database('AL-NI.tdb')
    >>> comps = ['AL', 'NI', 'VA']
    >>> phases = ['FCC_L12', 'BCC_B2', 'LIQUID', 'AL3NI5', 'AL3NI2', 'AL3NI']
    >>> callables = build_callables(dbf, comps, phases)
    >>> equilibrium(dbf, comps, phases, conditions, **callables)
    """
    comps = sorted(unpack_components(dbf, comps))
    pure_elements = get_pure_elements(dbf, comps)

    callables = {
        'massfuncs': {},
        'massgradfuncs': {},
        'callables': {},
        'grad_callables': {},
        'hess_callables': {},
    }

    models = unpack_kwarg(model, default_arg=Model)
    param_symbols = param_symbols if param_symbols is not None else []
    # wrap param symbols in Symbols if they are strings
    if all([isinstance(sym, string_types) for sym in param_symbols]):
        param_symbols = [Symbol(sym) for sym in param_symbols]
    param_values = np.zeros_like(param_symbols, dtype=np.float64)

    phase_records = {}
    # create models
    # starting from pycalphad
    for name in phases:
        mod = models[name]
        if isinstance(mod, type):
            models[name] = mod = mod(dbf, comps, name)
        site_fracs = mod.site_fractions
        variables = sorted(site_fracs, key=str)
        try:
            out = getattr(mod, output)
        except AttributeError:
            raise AttributeError('Missing Model attribute {0} specified for {1}'
                                 .format(output, mod.__class__))

        # Build the callables of the output
        # Only force undefineds to zero if we're not overriding them
        undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable) - set(param_symbols))
        for undef in undefs:
            out = out.xreplace({undef: float(0)})
        # XXX: Treating P and T special
        build_output = build_functions(out, tuple([v.P, v.T] + site_fracs), parameters=param_symbols,
                                       include_grad=build_gradients)
        if build_gradients:
            cf, gf = build_output
        else:
            cf = build_output
            gf = None
        hf = None
        callables['callables'][name] = cf
        callables['grad_callables'][name] = gf
        callables['hess_callables'][name] = hf

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
        callables['massfuncs'][name] = mcf
        callables['massgradfuncs'][name] = mgf

        # creating the phase records triggers the compile
        phase_records[name.upper()] = PhaseRecord_from_cython(comps, variables,
                                                              np.array(dbf.phases[name].sublattices, dtype=np.float),
                                                              param_values, callables['callables'][name],
                                                              callables['grad_callables'][name],
                                                              callables['hess_callables'][name],
                                                              callables['massfuncs'][name],
                                                              callables['massgradfuncs'][name])

    # finally, add the models to the callables
    callables['model'] = dict(models)
    return callables
