# -*- coding: utf-8 -*-
"""
Script for generating equations defining a semiprime factorization

@author: Richard
"""

import itertools
import math

import sympy
from semiprime_tools import num_to_factor_num_qubit
from sympy_helper_fns import max_value, is_equation


def generate_carry_equations(num_dig1=None, num_dig2=None, product=None):
    ''' Generate the carry equations for a given factorisation

        >>> product = 25
        >>> eqns = generate_carry_equations(product=product)
        >>> for e in eqns: print e
        p1 + q1 == 2*z12
        p1*q1 + z12 + 2 == 2*z23 + 4*z24
        p1 + q1 + z23 == 2*z34 + 1
        z24 + z34 + 1 == 2*z45 + 1
        z45 == 0

        >>> product = 143
        >>> eqns = generate_carry_equations(product=product)
        >>> for e in eqns: print e
        p1 + q1 == 2*z12 + 1
        p1*q1 + p2 + q2 + z12 == 2*z23 + 4*z24 + 1
        p1*q2 + p2*q1 + z23 + 2 == 2*z34 + 4*z35 + 1
        p1 + p2*q2 + q1 + z24 + z34 == 2*z45 + 4*z46
        p2 + q2 + z35 + z45 == 2*z56 + 4*z57
        z46 + z56 + 1 == 2*z67
        z57 + z67 == 2*z78 + 1
        z78 == 0
    '''
    if product is None:
        raise ValueError('generate_carry_equations must be given a product')
    if num_dig1 is None:
        assert num_dig2 is None
        num_dig1, num_dig2 = num_to_factor_num_qubit(product)

    eqns_rhs = [int(digit) for digit in bin(product)[2:][::-1]]
    eqns_lhs = [0 for _ in eqns_rhs]

    # Now pad them
    for i in xrange(5):
        eqns_lhs.append(0)
        eqns_rhs.append(0)

    ## Now add the contributions from the actual factors
    for pi in xrange(num_dig1):
        if pi in [0, num_dig1 - 1]:
            pi_str = '1'
        else:
            pi_str = 'p{}'.format(pi)
        for qi in xrange(num_dig2):
            if qi in [0, num_dig2 - 1]:
                qi_str = '1'
            else:
                qi_str = 'q{}'.format(qi)

            pq_str = '*'.join([pi_str, qi_str])
            eqns_lhs[pi + qi] += sympy.sympify(pq_str)

    ## Now loop over and add the carry variables
    for column_ind, sum_ in enumerate(eqns_lhs):
        if sum_ == 0:
            max_val = 1
        else:
            max_val = max_value(sum_)
        max_pow_2 = int(math.floor(math.log(max_val, 2)))
        for i in xrange(1, max_pow_2 + 1):
            z = sympy.Symbol('z{}{}'.format(column_ind, column_ind + i))
            eqns_rhs[column_ind] += (2 ** i) * z
            eqns_lhs[column_ind + i] += z

    eqns = [sympy.Eq(lhs, rhs) for lhs, rhs in zip(eqns_lhs, eqns_rhs)]
    eqns = filter(is_equation, eqns)

    return eqns

def generate_carry_equations_raw(num_dig1=None, num_dig2=None, product=None):
    ''' Generate the carry equations for a given factorisation

        >>> product = 25
        >>> eqns = generate_carry_equations_raw(product=product)
        >>> for e in eqns: print e
        p0*q0 == 1
        p0*q1 + p1*q0 == 2*z12
        p0*q2 + p1*q1 + p2*q0 + z12 == 2*z23 + 4*z24
        p1*q2 + p2*q1 + z23 == 2*z34 + 1
        p2*q2 + z24 + z34 == 2*z45 + 1
        z45 == 0

        >>> product = 143
        >>> eqns = generate_carry_equations_raw(product=product)
        >>> for e in eqns: print e
        p0*q0 == 1
        p0*q1 + p1*q0 == 2*z12 + 1
        p0*q2 + p1*q1 + p2*q0 + z12 == 2*z23 + 4*z24 + 1
        p0*q3 + p1*q2 + p2*q1 + p3*q0 + z23 == 2*z34 + 4*z35 + 1
        p1*q3 + p2*q2 + p3*q1 + z24 + z34 == 2*z45 + 4*z46
        p2*q3 + p3*q2 + z35 + z45 == 2*z56 + 4*z57
        p3*q3 + z46 + z56 == 2*z67
        z57 + z67 == 2*z78 + 1
        z78 == 0
    '''
    if product is None:
        raise ValueError('generate_carry_equations must be given a product')
    if num_dig1 is None:
        assert num_dig2 is None
        num_dig1, num_dig2 = num_to_factor_num_qubit(product)

    eqns_rhs = [int(digit) for digit in bin(product)[2:][::-1]]
    eqns_lhs = [0 for _ in eqns_rhs]

    # Now pad them
    for i in xrange(5):
        eqns_lhs.append(0)
        eqns_rhs.append(0)

    ## Now add the contributions from the actual factors
    for pi in xrange(num_dig1):
        pi_str = 'p{}'.format(pi)
        for qi in xrange(num_dig2):
            qi_str = 'q{}'.format(qi)

            pq_str = '*'.join([pi_str, qi_str])
            eqns_lhs[pi + qi] += sympy.sympify(pq_str)

    ## Now loop over and add the carry variables
    for column_ind, sum_ in enumerate(eqns_lhs):
        if sum_ == 0:
            max_val = 1
        else:
            max_val = max_value(sum_)
        max_pow_2 = int(math.floor(math.log(max_val, 2)))
        for i in xrange(1, max_pow_2 + 1):
            z = sympy.Symbol('z{}{}'.format(column_ind, column_ind + i))
            eqns_rhs[column_ind] += (2 ** i) * z
            eqns_lhs[column_ind + i] += z

    eqns = [sympy.Eq(lhs, rhs) for lhs, rhs in zip(eqns_lhs, eqns_rhs)]
    eqns = filter(is_equation, eqns)

    return eqns


def generate_factorisation_equation(num_dig1=None, num_dig2=None, product=None):
    ''' Generate the carry equations for a given factorisation

        >>> product = 9
        >>> eqn = generate_factorisation_equation(product=product)
        >>> print eqn
        p0*q0 + 2*p0*q1 + 2*p1*q0 + 4*p1*q1 == 9

        >>> product = 143
        >>> eqn = generate_factorisation_equation(product=product)
        >>> for i in xrange(0, len(str(eqn)), 80): print str(eqn)[i:i+80]
        p0*q0 + 2*p0*q1 + 4*p0*q2 + 8*p0*q3 + 2*p1*q0 + 4*p1*q1 + 8*p1*q2 + 16*p1*q3 + 4
        *p2*q0 + 8*p2*q1 + 16*p2*q2 + 32*p2*q3 + 8*p3*q0 + 16*p3*q1 + 32*p3*q2 + 64*p3*q
        3 == 143
    '''
    if product is None:
        raise ValueError('generate_carry_equations must be given a product')

    if num_dig1 is None:
        assert num_dig2 is None
        num_dig1, num_dig2 = num_to_factor_num_qubit(product)

    eqn_rhs = sympy.sympify(int(bin(product), 2))
    eqn_lhs = 0

    ## Now add the contributions from the actual factors
    for pi in xrange(num_dig1):
        for qi in xrange(num_dig2):
            pq_str = 'p{} * q{}'.format(pi, qi)
            eqn_lhs += sympy.sympify(pq_str) * 2 ** (pi + qi)

    return sympy.Eq(eqn_lhs, eqn_rhs)

def generate_carry_equations_auxiliary(num_dig1=None, num_dig2=None, product=None):
    ''' Given a product, generate the carry equations that express this
        using auxiliary variables for the p_i * q_j interactions

        >>> product = 25
        >>> eqns = generate_carry_equations_auxiliary(product=product)
        >>> for e in eqns: print e
        m0_0 == 1
        m0_1 + m1_0 == 2*z12
        m0_2 + m1_1 + m2_0 + z12 == 2*z23 + 4*z24
        m1_2 + m2_1 + z23 == 2*z34 + 1
        m2_2 + z24 + z34 == 2*z45 + 1
        z45 == 0
        p0*q0 == m0_0
        p0*q1 == m0_1
        p0*q2 == m0_2
        p1*q0 == m1_0
        p1*q1 == m1_1
        p1*q2 == m1_2
        p2*q0 == m2_0
        p2*q1 == m2_1
        p2*q2 == m2_2

        >>> product = 143
        >>> eqns = generate_carry_equations_auxiliary(product=product)
        >>> for e in eqns: print e
        m0_0 == 1
        m0_1 + m1_0 == 2*z12 + 1
        m0_2 + m1_1 + m2_0 + z12 == 2*z23 + 4*z24 + 1
        m0_3 + m1_2 + m2_1 + m3_0 + z23 == 2*z34 + 4*z35 + 1
        m1_3 + m2_2 + m3_1 + z24 + z34 == 2*z45 + 4*z46
        m2_3 + m3_2 + z35 + z45 == 2*z56 + 4*z57
        m3_3 + z46 + z56 == 2*z67
        z57 + z67 == 2*z78 + 1
        z78 == 0
        p0*q0 == m0_0
        p0*q1 == m0_1
        p0*q2 == m0_2
        p0*q3 == m0_3
        p1*q0 == m1_0
        p1*q1 == m1_1
        p1*q2 == m1_2
        p1*q3 == m1_3
        p2*q0 == m2_0
        p2*q1 == m2_1
        p2*q2 == m2_2
        p2*q3 == m2_3
        p3*q0 == m3_0
        p3*q1 == m3_1
        p3*q2 == m3_2
        p3*q3 == m3_3
    '''
    if product is None:
        raise ValueError('generate_carry_equations must be given a product')
    if num_dig1 is None:
        assert num_dig2 is None
        num_dig1, num_dig2 = num_to_factor_num_qubit(product)

    eqns_rhs = [int(digit) for digit in bin(product)[2:][::-1]]
    eqns_lhs = [0 for _ in eqns_rhs]

    # Create a holder for the equations that constrain pi.qj=mi_j
    constraints = []

    # Now pad them
    for i in xrange(5):
        eqns_lhs.append(0)
        eqns_rhs.append(0)

    ## Now add the contributions from the actual factors
    for pi in xrange(num_dig1):
        pi_str = 'p{}'.format(pi)
        for qi in xrange(num_dig2):
            qi_str = 'q{}'.format(qi)

            # Add the single interaction variable
            interaction_str = 'm{}_{}'.format(pi, qi)
            interaction_var = sympy.sympify(interaction_str)
            eqns_lhs[pi + qi] += interaction_var

            constraint = sympy.Eq(sympy.sympify('{}*{}'.format(pi_str, qi_str)),
                                  interaction_var)
            constraints.append(constraint)


    ## Now loop over and add the carry variables
    for column_ind, sum_ in enumerate(eqns_lhs):
        if sum_ == 0:
            max_val = 1
        else:
            max_val = max_value(sum_)
        max_pow_2 = int(math.floor(math.log(max_val, 2)))
        for i in xrange(1, max_pow_2 + 1):
            z = sympy.Symbol('z{}{}'.format(column_ind, column_ind + i))
            eqns_rhs[column_ind] += (2 ** i) * z
            eqns_lhs[column_ind + i] += z

    eqns = [sympy.Eq(lhs, rhs) for lhs, rhs in zip(eqns_lhs, eqns_rhs)]
    eqns = filter(is_equation, eqns)

    return eqns + constraints


FACTOR_ROOTS = ('p', 'q')


def cmp(a, b):
    return (a > b) - (a < b)

def cmp_variables(x, y):
    ''' Given 2 variables, compare them according to root (first letter) and
        then index.
        >>> variables = 'p1, p2, p10, p11, p22, q1, q10, q9, z12'.split(', ')
        >>> sorted(variables, cmp=cmp_variables)
        ['p1', 'p2', 'p10', 'p11', 'p22', 'q1', 'q9', 'q10', 'z12']
    '''
    x, y = str(x), str(y)
    if x[0] != y[0]:
        return cmp(x[0], y[0])
    else:
        return cmp(int(x[1:]), int(y[1:]))

def cmp_variables_right(x, y):
    ''' Given 2 variables, compare them according to whether their root is a
        factor, and then according to the power of 2 the variable represents.
        Low powers of 2 are given priority
        >>> variables = 'p1, p2, p10, p11, p22, q1, q10, q9, z12'.split(', ')
        >>> sorted(variables, cmp=cmp_variables_right)
        ['p1', 'q1', 'p2', 'q9', 'p10', 'q10', 'p11', 'p22', 'z12']
    '''
    x, y = str(x), str(y)
    x_root = x[0]
    y_root = y[0]
    x_ind = int(x[1:])
    y_ind = int(y[1:])

    # Now force p and q to be the same
    if (x_root in FACTOR_ROOTS) and (y_root in FACTOR_ROOTS):
        x_root = y_root = 'p'

    if x_root != y_root:
        return cmp(x_root, y_root)
    else:
        return cmp(x_ind, y_ind)

def cmp_variables_left(x, y):
    ''' Given 2 variables, compare them according to whether their root is a
        factor, and then according to the power of 2 the variable represents.
        High powers of 2 are given priority
        >>> variables = 'p1, p2, p10, p11, p22, q1, q10, q9, z12'.split(', ')
        >>> sorted(variables, cmp=cmp_variables_left)
        ['p22', 'p11', 'p10', 'q10', 'q9', 'p2', 'p1', 'q1', 'z12']
    '''
    x, y = str(x), str(y)
    x_root = x[0]
    y_root = y[0]
    x_ind = int(x[1:])
    y_ind = int(y[1:])

    # Now force p and q to be the same
    if (x_root in FACTOR_ROOTS) and (y_root in FACTOR_ROOTS):
        x_root = y_root = 'p'

    if x_root != y_root:
        return cmp(x_root, y_root)
    else:
        return -1 * cmp(x_ind, y_ind)

def factorisation_summary(prod):
    ''' Print a summary of what's going on.
        >>> print factorisation_summary(143)
        Hamming distance: 2
        10001111 =
        1101 x
        1011
        >>> print factorisation_summary(536275525501)
        Hamming distance: 11
        111110011011100100000110001111101111101 =
        10000000000000110101 x
        11111001101100101001
    '''
    # Hack around the circular import
    from verification import get_target_factors

    summary = []
    p, q = get_target_factors(prod)
    summary.append('Hamming distance: {}'.format(factor_binary_differences(p, q)))
    summary.append('{} =\n{} x\n{}'.format(bin(prod)[2:], bin(p)[2:], bin(q)[2:]))
    return '\n'.join(summary)

def factor_binary_differences(p, q):
    ''' Given 2 factors, work out how many places they differ by when expressed
        in binary form
        >>> for m, n in itertools.combinations_with_replacement(range(1, 10), 2):
        ...     if len(bin(m)) != len(bin(n)):
        ...         continue
        ...     print m, n, factor_binary_differences(m, n)
        1 1 0
        2 2 0
        2 3 1
        3 3 0
        4 4 0
        4 5 1
        4 6 1
        4 7 2
        5 5 0
        5 6 2
        5 7 1
        6 6 0
        6 7 1
        7 7 0
        8 8 0
        8 9 1
        9 9 0
        >>> factor_binary_differences(524309, 534167)
        5
        >>> factor_binary_differences(1048573, 1048423)
        4
        >>> factor_binary_differences(1267650600228229401496703205361, 633825993891935921676532842551)
        78
    '''
    p_str = bin(p)[2:]
    q_str = bin(q)[2:]
    if len(p_str) != len(q_str):
        return None
    diffs = 0
    for pi, qi in itertools.izip(p_str, q_str):
        if pi != qi:
            diffs += 1
    return diffs

def num_to_factor_num_qubit(prod):
    ''' Given a number, work out how many qubits the factors should be.
        Return the largest first
        >>> num_to_factor_num_qubit(143)
        (4, 4)
        >>> num_to_factor_num_qubit(56153)
        (8, 8)
        >>> num_to_factor_num_qubit(1099551473989)
        (21, 21)
        >>> num_to_factor_num_qubit(309485009822787627980424653)
        (45, 45)
        >>> num_to_factor_num_qubit(1267650600228508624673600186743)
        (51, 51)
    '''
    bin_str = bin(prod)[2:]
    num_qub = len(bin_str)
    if num_qub % 2:
        return (num_qub + 1) / 2, (num_qub + 1) / 2
    else:
        return num_qub / 2, num_qub / 2

if __name__ == "__main__":
    import doctest
    doctest.testmod()