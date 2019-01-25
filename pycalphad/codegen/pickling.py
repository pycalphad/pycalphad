import cloudpickle
from pycalphad.codegen.callables import build_callables

def dump_callables(dbf, pkl_fname, comps=None, phases=None, **kwargs):
    """
    Build and dump callables to a binary Pickle file

    Parameters
    ----------
    dbf : Database
        A Database object
    pkl_fname : str
        Path of a file to dump the callables.
    comps : list
        List of component names
    phases : list
        List of phase names
    kwargs : *
        Additional arguments to pass to build_callables.
        See the pycalphad.codegen.callables.build_callables docs.

    Returns
    -------
    None

    """
    comps = comps or sorted(dbf.elements - {'/-'})
    phases = phases or list(dbf.phases.keys())
    cbs = build_callables(dbf, comps, phases, **kwargs)
    with open(pkl_fname, 'wb') as fp:
        cloudpickle.dump(cbs, fp)

def load_callables(pkl_fname):
    """
    Load the callables at the specified path into a dictionary

    Parameters
    ----------
    pkl_fname :

    Returns
    -------
    dict
        Dictionary of keyword arguments to pass to `calculate` or `equilibrium`

    Examples
    --------
    >>> fname = 'my_callables.pkl'
    >>> eq_kwargs = load_callables(fname)  # doctest: +SKIP
    >>> equilibrium(dbf, comps, phases, conds, **eq_kwargs)  # doctest: +SKIP

    """
    with open(pkl_fname, 'rb') as fp:
        cbs = cloudpickle.load(fp)
    eq_kwargs = {
        'callables': cbs,
        'model': cbs['model']
    }
    return eq_kwargs
