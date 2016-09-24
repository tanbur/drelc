
import itertools

import sympy

from sympy_helper_fns import expressions_to_variables
from sympy_subs import subs


def local_search(interior_nodes, term_dict):
    ''' For every possible assignment of the variables not in interior_nodes, minimize the interior nodes and return
        the minimizing assignments.
    '''
    # Fix orders of variables
    exterior_nodes = tuple(term_dict.variables.difference(set(interior_nodes)))
    interior_nodes = tuple(interior_nodes)

    local_minima = []

    # For the moment, lets use sympy
    sympy_expr = term_dict.as_sympy_expr()
    for vals in itertools.product(range(2), repeat=len(exterior_nodes)):
        to_sub = dict(zip(exterior_nodes, vals))
        obj_func = subs(sympy_expr, to_sub)
        minima, value = minimize_sympy_expr(obj_func)
        local_minima.extend(minima)

    return local_minima


def minimize_sympy_expr_brute_force(sympy_expr):
    ''' Brute force minimization

        >>> expr = 'x1 + x2 - 2*x3'
        >>> print minimize_sympy_expr_brute_force(sympy.sympify(expr))
        [{x3: 1, x1: 0, x2: 0}]

        >>> expr = 'x1 + x2 - 2*x3*x1*x2'
        >>> print minimize_sympy_expr_brute_force(sympy.sympify(expr))
        [{x3: 0, x1: 0, x2: 0}, {x3: 1, x1: 0, x2: 0}, {x3: 1, x1: 1, x2: 1}]

        >>> expr = 'x1 + x2 - 3*x3*x1*x2'
        >>> print minimize_sympy_expr_brute_force(sympy.sympify(expr))
        [{x3: 1, x1: 1, x2: 1}]
    '''
    variables = expressions_to_variables([sympy_expr])
    min_value = None
    minima = []
    for vals in itertools.product(range(2), repeat=len(variables)):
        to_sub = dict(zip(variables, vals))
        val = subs(sympy_expr, to_sub)

        if min_value is None:
            min_value = val
            minima.append(to_sub)
            continue

        if val < min_value:
            min_value = val
            minima = [to_sub]
        elif val == min_value:
            minima.append(to_sub)
        else:
            continue
    return minima, min_value


def minimize_sympy_expr(sympy_expr):
    ''' Minimize a sympy expression the quickest way possible '''
    return minimize_sympy_expr_brute_force(sympy_expr)

if __name__ == '__main__':
    import doctest
    doctest.testmod()