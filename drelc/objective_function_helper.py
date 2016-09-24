# -*- coding: utf-8 -*-
"""
Created on Wed Jan 07 14:35:08 2015

@author: Richard
"""

from collections import defaultdict
import itertools
import operator

import sympy
from sympy.core.cache import clear_cache

from sympy_helper_fns import (remove_binary_squares, expressions_to_variables,
                              degree, num_add_terms, min_value, max_value,
                              str_eqns_to_sympy_eqns)
from sympy_subs import subs, subs_many
from term_dict import TermDict, SympyVariables

### Vanilla obj func
def eqn_to_vanilla_term_dict(eqn):
    ''' Take an equation and square it, throwing away any exponents
        >>> eqn = sympy.Eq(sympy.sympify('x - 1'))
        >>> term_dict = eqn_to_vanilla_term_dict(eqn)
        >>> term_dict
        defaultdict(<type 'int'>, {1: 1, x: -1})
    '''
    def _combine_terms((term1, term2)):
        ''' Small helper to recombine things

            >>> assert False
        '''
        term1, coef1 = term1
        term2, coef2 = term2

        atoms = term1.atoms().union(term2.atoms())
        atoms = sympy.Mul.fromiter(atoms)

        return atoms, coef1 * coef2

    terms = itertools.chain(eqn.lhs.as_coefficients_dict().iteritems(),
                                (-eqn.rhs).as_coefficients_dict().iteritems())
    products = itertools.product(terms, repeat=2)
    term_to_coef = itertools.imap(_combine_terms, products)

    # Now add up and return the term dict
    final_terms = TermDict(int)
    for term, coef in term_to_coef:
        final_terms[SympyVariables(term)] += coef
    return final_terms

def equations_to_vanilla_objective_function(equations):
    ''' Take a list of sympy equations and return an objective function

        >>> lhs = 'x'
        >>> rhs = '1'
        >>> eqns = [sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))]
        >>> equations_to_vanilla_objective_function(eqns)
        -x + 1

        >>> lhs = 'x + y'
        >>> rhs = 'x*y'
        >>> eqns = [sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))]
        >>> equations_to_vanilla_objective_function(eqns)
        -x*y + x + y
    '''
    # Watch out for edge cases
    if len(equations) == 0:
        return sympy.sympify('0')

    tds = map(eqn_to_vanilla_term_dict, equations)
    term_dict = sum_term_dicts(tds)
    return sum([k*v for k, v in term_dict.iteritems()])



def equations_to_vanilla_term_dict(equations):
    ''' Return the coefficient string of the objective function of some equations
        Also include the dictionary of variable number to original variable

        >>> lhs = 'x'
        >>> rhs = '1'
        >>> eqns = [sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))]
        >>> print equations_to_vanilla_term_dict(eqns)
        defaultdict(<type 'int'>, {1: 1, x: -1})

        >>> lhs = 'x + y'
        >>> rhs = 'x*y'
        >>> eqns = [sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))]
        >>> print equations_to_vanilla_term_dict(eqns)
        defaultdict(<type 'int'>, {x*y: -1, x: 1, y: 1})
    '''
    tds = map(eqn_to_vanilla_term_dict, equations)
    term_dict = sum_term_dicts(tds)
    return term_dict


def equations_to_vanilla_coef_str(equations):
    ''' Return the coefficient string of the objective function of some equations
        Also include the dictionary of variable number to original variable

        >>> lhs = 'x'
        >>> rhs = '1'
        >>> eqns = [sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))]
        >>> print equations_to_vanilla_coef_str(eqns)
        1
        1 -1
        <BLANKLINE>
        {x: 1}

        >>> lhs = 'x + y'
        >>> rhs = 'x*y'
        >>> eqns = [sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))]
        >>> print equations_to_vanilla_coef_str(eqns)
        1 1
        2 1
        1 2 -1
        <BLANKLINE>
        {x: 1, y: 2}
    '''
    term_dict = equations_to_vanilla_term_dict(equations)
    return term_dict_to_coef_string(term_dict)

### General helpers

def evaluate_term_dict(target_solns, term_dict):
    ''' Given a term dict, check that it evaluates to 0

        >>> from cfg_sympy_solver import EXPERIMENTS
        >>> from solver_judgement import SolverJudgement as SOLVER
        >>> from carry_equations_generator import generate_carry_equations
        >>> from verification import get_target_solutions

        >>> params = EXPERIMENTS[1][:3]

        >>> prod = params[-1]
        >>> eqns = generate_carry_equations(*params)
        >>> system = SOLVER(eqns)
        >>> system.solve_equations(max_iter=2)
        >>> term_dict = equations_to_vanilla_term_dict(system.equations)
        >>> evaluate_term_dict(get_target_solutions(prod), term_dict)
        0

        >>> params = EXPERIMENTS[2][:3]

        >>> prod = params[-1]
        >>> eqns = generate_carry_equations(*params)
        >>> system = SOLVER(eqns)
        >>> system.solve_equations(max_iter=2)
        >>> term_dict = equations_to_vanilla_term_dict(system.equations)
        >>> evaluate_term_dict(get_target_solutions(prod), term_dict)
        0
    '''
    terms = [coef*variables for (variables, coef) in term_dict.iteritems()]
    subbed = subs_many(terms, target_solns)
    return sum(subbed)

def sum_term_dicts(term_dicts):
    ''' Combine term dicts '''
    term_dict = TermDict(int)
    for t_d in term_dicts:
        for term, coef in t_d.iteritems():
            term_dict[term] += coef
    return term_dict

def eqns_to_exprs(eqn):
    ''' Take an equation, return lhs - rhs '''
    return eqn.lhs - eqn.rhs

def expressions_to_term_dict(exprs, process_eqn=None):
    ''' Given a list of Sympy expressions, calculate the term: coefficient dict
        of the objective function
    '''
    final_terms = TermDict(int)
    for i, expr in enumerate(exprs):
        for t, c in expr.as_coefficients_dict().iteritems():
            final_terms[SympyVariables(t)] += c
    return final_terms

def _assign_atoms_to_index(atoms):
    ''' Take an iterable of atoms and return an {atom: integer id} map

        >>> atoms = sympy.sympify('x1*x2*y1*z*a').atoms()
        >>> _assign_atoms_to_index(atoms)
        {x2: 3, y1: 4, x1: 2, a: 1, z: 5}
    '''
    atoms = sorted(atoms, key=str)
    atom_map = {v : i + 1 for i, v in enumerate(atoms)}
    return atom_map

def term_dict_to_coef_string(term_dict):
    ''' Given a dictionary of terms to coefficient, return the coefficient
        string for input into the optimisation code
    '''
    if not len(term_dict):
        return ''

    atoms = set.union(*[term.as_sympy_expr().atoms(sympy.Symbol) for term in term_dict.iterkeys()])
    atom_map = _assign_atoms_to_index(atoms)

    lines = []
    for term, coef in term_dict.iteritems():
        var_num = sorted([atom_map[atom] for atom in term.as_sympy_expr().atoms(sympy.Symbol)])
        line = ' '.join(map(str, var_num))
        if line:
            line += ' ' + str(coef)
        else:
            if coef == 0:
                continue
            line = str(coef)
        lines.append(line)

    lines = sorted(lines, key=len)

    coef_str = '\n'.join(lines)
    atom_str = str(atom_map)
    return '\n\n'.join([coef_str, atom_str])

def expression_to_coef_string(expr):
    ''' Return the coefficient string for a sympy expression
        Also include the dictionary of variable number to original variable

        >>> inp = '40*s_1 + 30*s_1*s_2 + 100*s_1*s_2*s_3 - 15*s_2*s_3 - 20*s_3 + 4'
        >>> print expression_to_coef_string(inp)
        4
        1 40
        3 -20
        1 2 30
        2 3 -15
        1 2 3 100
        <BLANKLINE>
        {s_3: 3, s_2: 2, s_1: 1}
    '''
    if isinstance(expr, str):
        expr = sympy.sympify(expr)

    return term_dict_to_coef_string(expr.as_coefficients_dict())

def coef_str_to_file(coef_str, filename=None):
    ''' Write the objective function to a file, or printing if None.
        Also include the dictionary of variable number to original variable
    '''
    if filename is None:
        print coef_str
    else:
        f = open(filename, 'a')
        f.write(coef_str)
        f.close()

def count_qubit_interactions(term_dict):
    ''' Given a term dict, count the number of terms that have an n-qubit
        interaction, and return the interaction profile
    '''
    count = defaultdict(int)
    for k, v in term_dict.iteritems():
        if v != 0:
            count[degree(k)] += 1
    return count

### Schaller stuff

## Original Schaller implementation when the equations come transformed, we
## just need to add them up
def equations_to_sum_coef_str(eqns):
    ''' Take equations and sum them up
        >>> equations = ['a + b - c*d - 1', 'x1*x2 + 3*x2*x3 - 2']
        >>> equations = map(sympy.sympify, equations)
        >>> equations = map(sympy.Eq, equations)
        >>> print equations_to_sum_coef_str(equations)
        -3
        1 1
        2 1
        6 7 3
        5 6 1
        3 4 -1
        <BLANKLINE>
        {x3: 7, c: 3, x2: 6, d: 4, x1: 5, a: 1, b: 2}
    '''
    exprs = map(eqns_to_exprs, eqns)
    term_dict = expressions_to_term_dict(exprs)
    return term_dict_to_coef_string(term_dict)


### Introduce auxilary variables
def schaller_simple((ab, s)):
    ''' Change (ab-s)**2 to (ab - sa - sb + s), which has minimums at exactly
        the point ab = s

        >>> vars_ = sympy.symbols('a b s')
        >>> a, b, s = vars_
        >>> orig_expr = (a*b - s)**2
        >>> schaller_expr = schaller_simple((a*b, s))

        >>> print schaller_expr
        a*b - 2*a*s - 2*b*s + 3*s

        >>> abss = itertools.product(range(2), repeat=3)
        >>> for abs in abss:
        ...     to_sub = dict(zip(vars_, abs))
        ...     if orig_expr.subs(to_sub) == 0:
        ...         assert schaller_expr.subs(to_sub) == 0
        ...     else:
        ...         assert schaller_expr.subs(to_sub) > 0
    '''
    assert len(ab.atoms(sympy.Symbol)) == 2
    a, b = ab.atoms(sympy.Symbol)
    return a*b - 2*a*s - 2*b*s + 3*s

def exprs_to_auxillary_term_dict(exprs):
    ''' Take equations and replace 2-qubit interactions with new variables

        >>> a, b, c, d, e = sympy.symbols('a b c d e')
        >>> eqns = [sympy.Eq(a, b), sympy.Eq(c*d, e)]
        >>> exprs = map(lambda x: x.lhs - x.rhs, eqns)
        >>> exprs_to_auxillary_term_dict(exprs)
        defaultdict(<type 'int'>, {a*b: -2, c*c_d: -2, c_d: 4, a: 1, c_d*e: -2, e: 1, b: 1, c_d*d: -2, c*d: 1})
    '''
    # First replace all 2 qubit terms with new variables and sub them in
    aux_s = {}
    new_exprs = []
    for expr in exprs:
        new_expr = 0
        for term in expr.as_ordered_terms():
            num_qubits = len(term.atoms(sympy.Symbol))
            if num_qubits < 2:
                new_expr += term
            elif num_qubits == 2:
                coef, qubits = term.as_coeff_Mul()
                s = sympy.Symbol('{}_{}'.format(*sorted(qubits.atoms(), key=str)))
                aux_s[qubits] = s
                new_expr += coef * s
            else:
                raise ValueError('{} too complex'.format(expr))
                new_expr += term
        new_exprs.append(new_expr)
    exprs = new_exprs
    #exprs = [expr.subs(aux_s) for expr in exprs]

    # Now take the squares ready for the final equation
    #exprs = [(expr**2).expand() for expr in exprs]
    exprs = map(sympy.Eq, exprs)
    tds = map(eqn_to_vanilla_term_dict, exprs)
    term_dict = sum_term_dicts(tds)

    # Now use the funky formula to provide equality for the auxillary variables
    aux_exprs = map(schaller_simple, aux_s.iteritems())
    for term, coef in expressions_to_term_dict(aux_exprs).iteritems():
        term_dict[term] += coef

    return term_dict

def equations_to_auxillary_coef_str(eqns):
    ''' Take equations and make the objective function nice

        >>> a, b, c, d, e = sympy.symbols('a b c d e')
        >>> eqns = [sympy.Eq(a, b), sympy.Eq(c*d, e)]
        >>> print equations_to_auxillary_coef_str(eqns)
        4 4
        6 1
        1 1
        2 1
        3 5 1
        1 2 -2
        3 4 -2
        4 5 -2
        4 6 -2
        <BLANKLINE>
        {c: 3, d: 5, a: 1, e: 6, c_d: 4, b: 2}
    '''
    exprs = map(lambda x: x.lhs - x.rhs, eqns)
    term_dict = exprs_to_auxillary_term_dict(exprs)
    return term_dict_to_coef_string(term_dict)


### Deduction reduction
def reduce_term_dict_with_deductions(term_dict, deductions, lagrangian_coefficient=0,
                                     preserve_terms=False, substitute_deductions=True):
    ''' Given a term dict and some deductions, simplify the term dict

        lagrangian_coefficient determines the extra additive coefficient in front of the
        Lagrangian multiplier of each deduction

        If preserve_terms is True then don't add any terms unless the
        term dict already contains the terms in the associated Lagrangian
        multiplier

        If substitute_deductions is True, try and substitute the deductions
        into terms already in the term dict. Useful for debugging or for quick
        runs

        >>> from collections import defaultdict
        >>> import sympy
        >>> term_dict = TermDict(int)
        >>> a, b, c, u, v, x, y, z = sympy.symbols('a, b, c, u, v, x, y, z')
        >>> term_dict[a*b*c] = 4
        >>> term_dict[u*v*z] = 8
        >>> term_dict[x*y*z] = 10

        >>> deductions = {
        ...     a*b: 0,
        ...     u*v: u + v - 1,
        ...     }

        >>> reduced = reduce_term_dict_with_deductions(term_dict, deductions,
        ...                            lagrangian_coefficient=0, preserve_terms=False)
        >>> for term, coef in reduced.iteritems(): print coef * term
        10*x*y*z
        4*a*b
        -8*v
        -8*u
        8*u*z
        8
        8*u*v
        -8*z
        8*v*z

        >>> reduced = reduce_term_dict_with_deductions(term_dict, deductions,
        ...                            lagrangian_coefficient=0, preserve_terms=True)
        >>> for term, coef in reduced.iteritems(): print coef * term
        10*x*y*z
        4*a*b*c
        8*u*v*z

        Setup
        >>> p1, q1, p2, q2, z1, z2 = sympy.symbols('p1 q1 p2 q2 z1 z2')
        >>> eqns = ['p1 + q1 == 1', 'p1*q2 + q1*p2 == z1 + 2*z2']
        >>> eqns = str_eqns_to_sympy_eqns(eqns)
        >>> term_dict1 = equations_to_vanilla_term_dict(eqns)
        >>> atoms = sorted(expressions_to_variables(eqns), key=str)

        Check xy = 0 judgement
        >>> term_dict2 = reduce_term_dict_with_deductions(term_dict1.copy(), {p1 * q1: 0})
        >>> for vals in itertools.product(range(2), repeat=len(atoms)):
        ...     to_sub = dict(zip(atoms, vals))
        ...     val1 = evaluate_term_dict(to_sub, term_dict1)
        ...     val2 = evaluate_term_dict(to_sub, term_dict2)
        ...     #print val1, val2
        ...     assert val1 >= 0
        ...     if val1 == 0:
        ...         assert val2 == 0
        ...     else:
        ...         assert val2 > 0

        Check xy = x + y - 1 judgement
        >>> term_dict2 = reduce_term_dict_with_deductions(term_dict1.copy(), {p1 * q1: p1 + q1 - 1})
        >>> for vals in itertools.product(range(2), repeat=len(atoms)):
        ...     to_sub = dict(zip(atoms, vals))
        ...     val1 = evaluate_term_dict(to_sub, term_dict1)
        ...     val2 = evaluate_term_dict(to_sub, term_dict2)
        ...     #print val1, val2
        ...     assert val1 >= 0
        ...     if val1 == 0:
        ...         assert val2 == 0
        ...     else:
        ...         assert val2 > 0

        Setup 2
        We need z1 = 1 to test the next kind of assumption
        >>> eqns = ['p1 + q1 == 1', 'p1*q2 + q1*p2 == 1 + 2*z2']
        >>> eqns = str_eqns_to_sympy_eqns(eqns)
        >>> term_dict1 = equations_to_vanilla_term_dict(eqns)
        >>> atoms = sorted(expressions_to_variables(eqns), key=str)

        Check xy = x judgement. First we need to fix z1=1 to assert this
        >>> term_dict2 = reduce_term_dict_with_deductions(term_dict1.copy(), {p1*q2: p1})
        >>> for vals in itertools.product(range(2), repeat=len(atoms)):
        ...     to_sub = dict(zip(atoms, vals))
        ...     val1 = evaluate_term_dict(to_sub, term_dict1)
        ...     val2 = evaluate_term_dict(to_sub, term_dict2)
        ...     #print val1, val2
        ...     assert val1 >= 0
        ...     if val1 == 0:
        ...         assert val2 == 0
        ...     else:
        ...         assert val2 > 0

    '''
    term_dict = term_dict.copy()
    for poly, value in deductions.iteritems():
        clear_cache()
#        assert degree(poly) > 1
        assert num_add_terms(poly) == 1

        # Work out what the Lagrangian multiplier is of the deduction. We might
        # not want to go any further
        constraint = remove_binary_squares(((poly - value)**2).expand())

        # Check if adding the Lagrangian would alter the term-dicts profile
        if preserve_terms:
            is_subset = True
            for term in constraint.as_coefficients_dict().keys():
                if term_dict[term] == 0:
                    is_subset = False
                    break
            if not is_subset:
                continue

        # Constraint coefficient is the multiplier for the error term
        constraint_coefficient = lagrangian_coefficient

        # If we want to, use the deductions to try and reduce existing terms
        if substitute_deductions:
            poly_atoms = poly.atoms(sympy.Symbol)
            for term, coef in term_dict.copy().iteritems():
                if poly_atoms.issubset(term.atoms(sympy.Symbol)):
                    # Remove the reference to the old term
                    term_dict.pop(term)

                    # Add on the new terms under the judgement
                    new_term = subs(term, {poly: value}) * coef
                    if not isinstance(new_term, int):
                        new_term = remove_binary_squares(new_term.expand())
    #                new_term1 = term.subs(poly, value).expand() * coef
    #                assert new_term1 == new_term
                    constraint_coefficient += max(coef, 0)
                    for _term, _coef in new_term.as_coefficients_dict().iteritems():
                        if _coef != 0:
                            term_dict[_term] += _coef

        # Now add multiples of the constraint^2 to make sure we are looking at
        # the same ground states. Multiply by the absolute value of the
        # coefficients to make sure no negative states occur
        for _term, _coef in constraint.as_coefficients_dict().iteritems():
            term_dict[_term] += _coef * constraint_coefficient

    return term_dict

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    a, b, c, d, e, f = sympy.symbols('a b c d e f')
    syms = [a, b, c, d, e, f]
    exprs = [a + b,
             a*b + c*d,
             a*b + b*c,
             a*b + c*d + e,
             a*b + c*d - 1
             ]

    new = sum([_t*_c for _t, _c in exprs_to_auxillary_term_dict(exprs).iteritems()])
    exprs = sum(map(lambda x: (x**2).expand(), exprs))

    for abcde in itertools.product(xrange(2), repeat=5):
        to_sub = zip(syms, abcde)
        orig_val = exprs.subs(to_sub)
        new = sympy.sympify(str(new).replace('_', '*'))
        new_val = new.subs(to_sub)
        if orig_val == 0:
            assert new_val == 0
        else:
            assert new_val > 0