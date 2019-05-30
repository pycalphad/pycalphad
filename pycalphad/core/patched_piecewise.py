"""
Copied from Piecewise SymPy. The only modification is in `piecewise_eval` where

```
    for e, c in _args:
        if not c.is_Atom and not isinstance(c, Relational):
            free = c.free_symbols
```

is changed to

```
    for e, c in _args:
        if not c.is_Atom and not isinstance(c, Relational):
            free = c.expr_free_symbols
```

See the following links:
https://github.com/sympy/sympy/issues/14933
https://github.com/pycalphad/pycalphad/pull/180

"""

import sympy.functions.elementary.piecewise
from sympy.core import S, Function, Dummy, Tuple
from sympy.core.basic import as_Basic
from sympy.core.relational import Relational, _canonical
from sympy.logic.boolalg import And, Boolean, distribute_and_over_or, Or, true, false
from sympy.utilities.misc import filldedent, func_name

# Removes ITE rewriting, which is not compatible with SymEngine
def exprcondpair_new(cls, expr, cond):
    expr = as_Basic(expr)
    if cond == True:
        return Tuple.__new__(cls, expr, true)
    elif cond == False:
        return Tuple.__new__(cls, expr, false)

    if not isinstance(cond, Boolean):
        raise TypeError(filldedent('''
            Second argument must be a Boolean,
            not `%s`''' % func_name(cond)))
    return Tuple.__new__(cls, expr, cond)

def piecewise_eval(cls, *_args):
    if not _args:
        return

    if len(_args) == 1 and _args[0][-1] == True:
        return _args[0][0]

    newargs = []  # the unevaluated conditions
    current_cond = set()  # the conditions up to a given e, c pair
    # make conditions canonical
    args = []
    for e, c in _args:
        if not c.is_Atom and not isinstance(c, Relational):
            free = c.expr_free_symbols
            if len(free) == 1:
                funcs = [i for i in c.atoms(Function)
                         if not isinstance(i, Boolean)]
                if len(funcs) == 1 and len(
                        c.xreplace({list(funcs)[0]: Dummy()}
                                   ).free_symbols) == 1:
                    # we can treat function like a symbol
                    free = funcs
                _c = c
                x = free.pop()
                try:
                    c = c.as_set().as_relational(x)
                except NotImplementedError:
                    pass
                else:
                    reps = {}
                    for i in c.atoms(Relational):
                        ic = i.canonical
                        if ic.rhs in (S.Infinity, S.NegativeInfinity):
                            if not _c.has(ic.rhs):
                                # don't accept introduction of
                                # new Relationals with +/-oo
                                reps[i] = S.true
                            elif ('=' not in ic.rel_op and
                                  c.xreplace({x: i.rhs}) !=
                                  _c.xreplace({x: i.rhs})):
                                reps[i] = Relational(
                                    i.lhs, i.rhs, i.rel_op + '=')
                    c = c.xreplace(reps)
        args.append((e, _canonical(c)))

    for expr, cond in args:
        # Check here if expr is a Piecewise and collapse if one of
        # the conds in expr matches cond. This allows the collapsing
        # of Piecewise((Piecewise((x,x<0)),x<0)) to Piecewise((x,x<0)).
        # This is important when using piecewise_fold to simplify
        # multiple Piecewise instances having the same conds.
        # Eventually, this code should be able to collapse Piecewise's
        # having different intervals, but this will probably require
        # using the new assumptions.
        if isinstance(expr, sympy.functions.elementary.piecewise.Piecewise):
            unmatching = []
            for i, (e, c) in enumerate(expr.args):
                if c in current_cond:
                    # this would already have triggered
                    continue
                if c == cond:
                    if c != True:
                        # nothing past this condition will ever
                        # trigger and only those args before this
                        # that didn't match a previous condition
                        # could possibly trigger
                        if unmatching:
                            expr = sympy.functions.elementary.piecewise.Piecewise(*(
                                    unmatching + [(e, c)]))
                        else:
                            expr = e
                    break
                else:
                    unmatching.append((e, c))

        # check for condition repeats
        got = False
        # -- if an And contains a condition that was
        #    already encountered, then the And will be
        #    False: if the previous condition was False
        #    then the And will be False and if the previous
        #    condition is True then then we wouldn't get to
        #    this point. In either case, we can skip this condition.
        for i in ([cond] +
                  (list(cond.args) if isinstance(cond, And) else
                  [])):
            if i in current_cond:
                got = True
                break
        if got:
            continue

        # -- if not(c) is already in current_cond then c is
        #    a redundant condition in an And. This does not
        #    apply to Or, however: (e1, c), (e2, Or(~c, d))
        #    is not (e1, c), (e2, d) because if c and d are
        #    both False this would give no results when the
        #    true answer should be (e2, True)
        if isinstance(cond, And):
            nonredundant = []
            for c in cond.args:
                if (isinstance(c, Relational) and
                        (~c).canonical in current_cond):
                    continue
                nonredundant.append(c)
            cond = cond.func(*nonredundant)
        elif isinstance(cond, Relational):
            if (~cond).canonical in current_cond:
                cond = S.true

        current_cond.add(cond)

        # collect successive e,c pairs when exprs or cond match
        if newargs:
            if newargs[-1].expr == expr:
                orcond = Or(cond, newargs[-1].cond)
                if isinstance(orcond, (And, Or)):
                    orcond = distribute_and_over_or(orcond)
                newargs[-1] = sympy.functions.elementary.piecewise.ExprCondPair(expr, orcond)
                continue
            elif newargs[-1].cond == cond:
                orexpr = Or(expr, newargs[-1].expr)
                if isinstance(orexpr, (And, Or)):
                    orexpr = distribute_and_over_or(orexpr)
                newargs[-1] = sympy.functions.elementary.piecewise.ExprCondPair(orexpr, cond)
                continue

        newargs.append(sympy.functions.elementary.piecewise.ExprCondPair(expr, cond))

    # some conditions may have been redundant
    missing = len(newargs) != len(_args)
    # some conditions may have changed
    same = all(a == b for a, b in zip(newargs, _args))
    # if either change happened we return the expr with the
    # updated args
    if not newargs:
        raise ValueError(filldedent('''
            There are no conditions (or none that
            are not trivially false) to define an
            expression.'''))
    if missing or not same:
        return cls(*newargs)
