# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:49:05 2015

@author: Richard
"""

import sympy

from ortools.constraint_solver import pywrapcp

from sympy_helper_fns import expressions_to_variables, str_eqns_to_sympy_eqns


def unique_array_stable(array):
    ''' Given a list of things, return a new list with unique elements with
        original order preserved (by first occurence)

        >>> print unique_array_stable([1, 3, 5, 4, 7, 4, 2, 1, 9])
        [1, 3, 5, 4, 7, 2, 9]
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in array if not (x in seen or seen_add(x))]

def parse_expression(expr, variable_lookup):
    ''' Given a string expression, form a new type of equation that or-tools
        will understand
    '''
    expr = str(expr)
    expr_ort = 0
    summands = expr.split('+')
    summands = [_s.strip() for _s in summands]
    for summand in summands:
        summand_ort = 1
        prods = summand.split('*')
        for prod in prods:
            try:
                summand_ort *= int(prod)
            except ValueError:
                summand_ort *= variable_lookup[prod]
        expr_ort += summand_ort
    return expr_ort


def str_solution_to_sympy_solution(str_solution):
    ''' Turn a printout from or-tools into one we can use in sympy

        >>> str_solution_to_sympy_solution('[x(0), y(1), z(0)]')
        {x: 0, z: 0, y: 1}

        >>> str_solution_to_sympy_solution('[p1(0), q2(1), z25(0)]')
        {z25: 0, p1: 0, q2: 1}
    '''
    str_solution = str_solution.strip('[]')
    str_solution = str_solution.replace(')','').replace('(', ':')
    solution = {}
    for assignment in str_solution.split(','):
        variable, value = assignment.strip().split(':')
        variable = sympy.Symbol(variable)
        solution[variable] = int(value)
    return solution

def minimize_expr(expr):
    ''' Given a sympy expression, minimize it

        >>> expr = 'x1 + x2 - 2*x3'
        >>> print minimize_expr(sympy.sympify(expr))
        [{x3: 1, x1: 0, x2: 0}]

        >>> expr = 'x1 + x2 - 2*x3*x1*x2'
        >>> print minimize_expr(sympy.sympify(expr))
        [{x3: 0, x1: 0, x2: 0}, {x3: 1, x1: 0, x2: 0}, {x3: 1, x1: 1, x2: 1}]

        >>> expr = 'x1 + x2 - 3*x3*x1*x2'
        >>> print minimize_expr(sympy.sympify(expr))
        [{x3: 1, x1: 1, x2: 1}]

    '''
    # # x and y are integer non-negative variables.
    # x = solver.IntVar(0, 17, 'x')
    # y = solver.IntVar(0, 17, 'y')
    # solver.Add(2 * x + 14 * y <= 35)
    # solver.Add(2 * x <= 7)
    # obj_expr = solver.IntVar(0, 1000, "obj_expr")
    # solver.Add(obj_expr == x + 10 * y)
    # objective = solver.Maximize(obj_expr, 1)
    # decision_builder = solver.Phase([x, y],
    #                                 solver.CHOOSE_FIRST_UNBOUND,
    #                                 solver.ASSIGN_MIN_VALUE)
    # # Create a solution collector.
    # collector = solver.LastSolutionCollector()
    # # Add the decision variables.
    # collector.Add(x)
    # collector.Add(y)
    # # Add the objective.
    # collector.AddObjective(obj_expr)
    # solver.Solve(decision_builder, [objective, collector])




    solver_params = pywrapcp.Solver.DefaultSolverParameters()
    solver = pywrapcp.Solver('enumerator', solver_params)

    # Extract the sympy variables, and create equivalents in or-tools
    variables_sympy = sorted(expressions_to_variables([expr]), key=str)
    variable_lookup = {}
    for variable in variables_sympy:
        variable_lookup[str(variable)] = solver.IntVar(0, 1, str(variable))

    variables_of_interest = set(variable_lookup.keys())

    # Create the objective function
    obj_expr = solver.IntVar(-10000, 100000, "obj_expr")

    # Now parse the equations
    expr_ort = parse_expression(expr, variable_lookup)
    # solver.Add(expr_ort == obj_expr)
    objective = solver.Minimize(expr_ort, 1)

    # Now order the variables so that we can perform clever searches
    variables = unique_array_stable(variables_of_interest)
    variables = [variable_lookup[v] for v in variables if v in variables_of_interest]

    # Perform the search
    db = solver.Phase(variables,
                      solver.CHOOSE_FIRST_UNBOUND,
                      solver.ASSIGN_MIN_VALUE)

    # Create a solution collector.
    collector = solver.LastSolutionCollector()
    # Add the decision variables.
    for v in variables:
        collector.Add(v)
    # Add the objective.
    collector.AddObjective(obj_expr)
    solver.Solve(db, [objective, collector])
    assert collector.SolutionCount()


    #### BLEH
    minima = []
    for min_ind in xrange(collector.SolutionCount()):
        minimum = {}

        collector.ObjectiveValue(best_solution))
        print()
        for v in variables:
            print('x= ', collector.Value(best_solution, v))

#     str_solutions = []
#     while solver.NextSolution():
# #        return [str_solution_to_sympy_solution(str(variables))]
#         str_solutions.append(str(variables))
# #        yield str_solution_to_sympy_solution(str(variables))
#
#     return map(str_solution_to_sympy_solution, str_solutions)

def enumerate_solutions(eqns, variables_of_interest=None, profile=False):
    ''' Given a set of equations, return every possible valid assignment

        Simple examples to see what ortools is made of

        1.
        >>> eqns = ['x + y + z == 1']
        >>> eqns = str_eqns_to_sympy_eqns(eqns)

        >>> sorted(enumerate_solutions(eqns, None), key=str)
        [{x: 0, z: 0, y: 1}, {x: 0, z: 1, y: 0}, {x: 1, z: 0, y: 0}]
        >>> sorted(enumerate_solutions(eqns, 'x'), key=str)
        [{x: 0}, {x: 1}]
        >>> sorted(enumerate_solutions(eqns, 'xy'), key=str)
        [{x: 0, y: 0}, {x: 0, y: 1}, {x: 1, y: 0}]


        2.
        >>> eqns = ['x + y + z == 3']
        >>> eqns = str_eqns_to_sympy_eqns(eqns)

        >>> enumerate_solutions(eqns, None)
        [{x: 1, z: 1, y: 1}]
        >>> enumerate_solutions(eqns, 'x')
        [{x: 1}]
        >>> enumerate_solutions(eqns, 'xy')
        [{x: 1, y: 1}]


        3. Parity - not so good
        >>> eqns = ['2*x1 + 4*x2 + y == 2*z1 + 4*z2 + 1']
        >>> eqns = str_eqns_to_sympy_eqns(eqns)

        >>> for sol in sorted(enumerate_solutions(eqns, None), key=str):
        ...     print sol
        {x1: 0, z2: 0, x2: 0, y: 1, z1: 0}
        {x1: 0, z2: 1, x2: 1, y: 1, z1: 0}
        {x1: 1, z2: 0, x2: 0, y: 1, z1: 1}
        {x1: 1, z2: 1, x2: 1, y: 1, z1: 1}
        >>> enumerate_solutions(eqns, 'y')
        [{y: 0}, {y: 1}]
        >>> sorted(enumerate_solutions(eqns, ['x1', 'y']), key=str)
        [{x1: 0, y: 1}, {x1: 1, y: 0}, {x1: 1, y: 1}]
    '''
    if not eqns:
        return []
    if profile:
        # Use some profiling and change the default parameters of the solver
        solver_params = pywrapcp.Solver.DefaultSolverParameters()

        # Change the profile level
        solver_params.profile_level = pywrapcp.SolverParameters.NORMAL_PROFILING

    else:
        solver_params = pywrapcp.Solver.DefaultSolverParameters()

    solver = pywrapcp.Solver('enumerator', solver_params)

    # Extract the sympy variables, and create equivalents in or-tools
    variables_sympy = sorted(expressions_to_variables(eqns), key=str)
    variable_lookup = {}
    for variable in variables_sympy:
        variable_lookup[str(variable)] = solver.IntVar(0, 1, str(variable))

    # Munge variables_of_interest
    if variables_of_interest is None:
        variables_of_interest = set(variable_lookup.keys())
    else:
        variables_of_interest = set(map(str, variables_of_interest))

    # Now parse the equations
    for eqn in eqns:
        solver.Add(parse_expression(eqn.lhs, variable_lookup) ==
                   parse_expression(eqn.rhs, variable_lookup))

    # Now order the variables so that we can perform clever searches
    variables = []
    for eqn in eqns:
        var_to_search = map(str, set(eqn.atoms(sympy.Symbol)))
        var_to_search = sorted(var_to_search, reverse=True)
        variables.extend(var_to_search)
    variables = unique_array_stable(variables)
    variables = [variable_lookup[v] for v in variables if v in variables_of_interest]

    # Perform the search
    db = solver.Phase(variables,
                      solver.CHOOSE_FIRST_UNBOUND,
                      solver.ASSIGN_MIN_VALUE)

    solver.NewSearch(db)

    str_solutions = []
    while solver.NextSolution():
#        return [str_solution_to_sympy_solution(str(variables))]
        str_solutions.append(str(variables))
#        yield str_solution_to_sympy_solution(str(variables))

    return map(str_solution_to_sympy_solution, str_solutions)

if __name__ == '__main__':
    import doctest
    doctest.testmod()