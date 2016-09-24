# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:37:05 2015

@author: Richard
"""

import sympy

from equivalence_dict import BinaryEquivalenceDict

from collections import defaultdict
import itertools

from sympy_helper_fns import is_simple_binary, str_eqns_to_sympy_eqns, degree

from contradiction_exception import ContradictionException

STATES_TO_JUDGEMENT_FINISHED = {

    # Anything empty means we've hit a contradiction
    tuple(): ContradictionException,

    # Linear
    ((0,),): lambda x: [(x, 0)],
    ((1,),): lambda x: [(x, 1)],
    ((0,), (1,)): lambda x: None,

    # Quadratic
    ((0, 0),): lambda x, y: [(x, 0), (y, 0)],
    ((1, 0),): lambda x, y: [(x, 1), (y, 0)],
    ((0, 1),): lambda x, y: [(x, 0), (y, 1)],
    ((1, 1),): lambda x, y: [(x, 1), (y, 1)],
    ((0, 0), (1, 1)): lambda x, y: [(x, y),],
    ((1, 0), (1, 1)): lambda x, y: [(x, 1),],
    ((0, 0), (1, 0)): lambda x, y: [(y, 0),],
    ((0, 1), (1, 1)): lambda x, y: [(y, 1),],
    ((0, 1), (1, 0)): lambda x, y: [(x, 1 - y),],
    ((0, 0), (0, 1)): lambda x, y: [(x, 0),],
    ((0, 0), (0, 1), (1, 0)): lambda x, y: [(x*y, 0),],
    ((0, 0), (1, 0), (1, 1)): lambda x, y: [(x*y, y),],
    ((0, 0), (0, 1), (1, 1)): lambda x, y: [(x*y, x),],
    ((0, 1), (1, 0), (1, 1)): lambda x, y: [(x*y, x + y - 1),],
    ((0, 0), (0, 1), (1, 0), (1, 1)): lambda x, y: None,

    # Qubic
    # Fully determined
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0),],
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1)): lambda x, y, z: [(x*y*z, x*y),],
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: [(x*y*z, x*z),],
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)): lambda x, y, z: [(x*y*z, x*z), (x*y, x*z),],
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0), (x*y, x - x*z)],  # Full formulation of (2*x*y*z, x*y + x*z - x)
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: [(x*y*z, x*y + x*z - x),],
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: [(x*y*z, y*z)],
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)): lambda x, y, z: [(x*y, y*z)],
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: [(x*z, y*z)],
    ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: [(x*y*z, y*z), (y*z, x*y + x*z - x)],
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0), (x*y, y - y*z), ],
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: [(x*y*z, x*y + y*z - y)],
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: [(x*y*z, x*z), (x*z, x*y + y*z - y)],
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0), (x*z, x + y - 2*x*y - y*z)],
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: [(x*y*z, x*y + y*z - y), (x*z, y*z + x - y),],
    ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0), (x*z, z - y*z)],
    ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: [(x*y*z,x*z + y*z - z)],
    ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1)): lambda x, y, z: [(x*y*z, x*y), (x*y, y*z + x*z - z)],
    ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: [(2*x*y*z, 2*x*z + x*y + y*z - x - z),],
    ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0), (2*x*y, x - z + y),],

}

# Partially determined
STATES_TO_JUDGEMENT_UNFINISHED = {
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 1)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 1), (0, 1, 0), (1, 0, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 1)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 1)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 0, 1)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 0)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 1)): lambda x, y, z: [(x*y*z, 0)],
	((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
	((0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)): lambda x, y, z: None,
}

STATES_TO_JUDGEMENT = {}
STATES_TO_JUDGEMENT.update(STATES_TO_JUDGEMENT_UNFINISHED)
STATES_TO_JUDGEMENT.update(STATES_TO_JUDGEMENT_FINISHED)

def dev_judgement_dict():
    '''
        >>> #dev_judgement_dict()
    '''
    for state_tuple, judgement in STATES_TO_JUDGEMENT.iteritems():
        if state_tuple == tuple():
            continue
        # Work out the invalid states, and the number of variables we are dealing with
        valid_states = set(state_tuple)
        num_var = len(next(iter(valid_states)))
        all_states = set(itertools.product(range(2), repeat=num_var))

        if len(valid_states) == 2**num_var:
            continue

        assert valid_states.issubset(all_states)
        invalid_states = all_states.difference(valid_states)

        # Check all the deductions in the valid states are true
        for state in valid_states:
            deductions = judgement(*state)
            if deductions is None:
                continue
            deduction_truth = [d[0] == d[1] for d in deductions]
            if not all(deduction_truth):
                print '1'
                print 'Valid states:\n', sorted(state_tuple)
                print 'Invalid states:\n', sorted(invalid_states)
                print 'Failed state:\n', state
                print 'Deductions:\n', deductions, deduction_truth
            assert all(deduction_truth)

        # And all the deductions for the invalid states are False
        for state in invalid_states:
            deductions = judgement(*state)
            if deductions is None:
                continue
            deduction_truth = [d[0] == d[1] for d in deductions]
            if all(deduction_truth):
                print '2'
                print 'Valid states:\n', sorted(state_tuple)
                print 'Invalid states:\n', sorted(invalid_states)
                print 'Failed state:\n', state
                print 'Deductions:\n', deductions, deduction_truth
                print
#            assert not all(deduction_truth)

def test_judgement_dict():
    ''' Test that all of the deductions make sense

        >>> test_judgement_dict()
    '''
    for state_tuple, judgement in STATES_TO_JUDGEMENT_FINISHED.iteritems():
        if state_tuple == tuple():
            continue
        # Work out the invalid states, and the number of variables we are dealing with
        valid_states = set(state_tuple)
        num_var = len(next(iter(valid_states)))
        all_states = set(itertools.product(range(2), repeat=num_var))
        assert valid_states.issubset(all_states)
        invalid_states = all_states.difference(valid_states)

        # Check all the deductions in the valid states are true
        for state in valid_states:
            deductions = judgement(*state)
            if deductions is None:
                continue
            deduction_truth = [d[0] == d[1] for d in deductions]
            assert all(deduction_truth)

        # And all the deductions for the invalid states are False
        for state in invalid_states:
            deductions = judgement(*state)
            if deductions is None:
                continue
            deduction_truth = [d[0] == d[1] for d in deductions]
#            if all(deduction_truth):
#                print sorted(state_tuple), state, deduction_truth, deductions

            assert not all(deduction_truth)


    for state_tuple, judgement in STATES_TO_JUDGEMENT_UNFINISHED.iteritems():
        if state_tuple == tuple():
            continue
        # Work out the invalid states, and the number of variables we are dealing with
        valid_states = set(state_tuple)
        num_var = len(next(iter(valid_states)))
        all_states = set(itertools.product(range(2), repeat=num_var))
        assert valid_states.issubset(all_states)
        invalid_states = all_states.difference(valid_states)

        # Check all the deductions in the valid states are true
        for state in valid_states:
            deductions = judgement(*state)
            if deductions is None:
                continue
            deduction_truth = [d[0] == d[1] for d in deductions]
            assert all(deduction_truth)

UNKNOWN_STATES = set()

def get_deduction_function(states):
    ''' Given a list of states, retrieve the function that gives the correct
        deduction


        >>> x, y, z = sympy.symbols('x y z')
        >>> print get_deduction_function(((0, 1), (1, 0)))(x, y)
        [(x, -y + 1)]

        >>> print get_deduction_function(((1, 0), (0, 1)))(x, y)
        [(x, -y + 1)]

        >>> print get_deduction_function(((0, 1, 0),))(x, y, z)
        ((x, 0), (y, 1), (z, 0))

    '''
    states = tuple(sorted(states))

    # If we only have 1 state, we can work that out on our own
    if len(states) == 1:
        return lambda *variables: tuple([(variable, value) for variable, value in
                                        itertools.izip(variables, states[0])])

    judgement_func = STATES_TO_JUDGEMENT.get(states)
    if judgement_func is None:
#        print 'Unknown combination {}'.format(states)
        UNKNOWN_STATES.add(states)
    return judgement_func

def _print_unknown_template():
    ''' Print out a template for unknown states, for convenience '''
    for u in sorted(UNKNOWN_STATES):
        print '\t{}: lambda x, y, z: None,'.format(u)

def aggregate_valid_states(valid_states, degree=2):
    ''' Given a list of valid substitution dicts, we aggregate all valid states
        according to max_degree

        >>> a, b, c, x, y, z = sympy.symbols('a b c x y z')
        >>> valid_states = [{a: 1, b: 0}, {a: 0, b: 1}, {a: 0, b: 0}]
        >>> for var_comb in aggregate_valid_states(valid_states, degree=1).iteritems(): print var_comb
        ((b,), ((0,), (1,)))
        ((a,), ((0,), (1,)))

        >>> for var_comb in aggregate_valid_states(valid_states, degree=2).iteritems(): print var_comb
        ((b, a), ((0, 0), (0, 1), (1, 0)))

        >>> valid_states = [{a: 1, b: 0, c: 1}, {a: 0, b: 1, c: 1}, {a: 0, b: 0, c: 1}]
        >>> for var_comb in aggregate_valid_states(valid_states, degree=1).iteritems(): print var_comb
        ((b,), ((0,), (1,)))
        ((c,), ((1,),))
        ((a,), ((0,), (1,)))

        >>> for var_comb in aggregate_valid_states(valid_states, degree=2).iteritems(): print var_comb
        ((b, a), ((0, 0), (0, 1), (1, 0)))
        ((c, a), ((1, 0), (1, 1)))
        ((c, b), ((1, 0), (1, 1)))

        >>> for var_comb in aggregate_valid_states(valid_states, degree=3).iteritems(): print var_comb
        ((c, b, a), ((1, 0, 0), (1, 0, 1), (1, 1, 0)))
    '''
    atoms = set.intersection(*map(set, (state.keys() for state in valid_states)))

    variable_combinations = {}

    for variables in itertools.combinations(atoms, r=degree):
        variable_combinations[variables] = set()

    for variables in variable_combinations.iterkeys():
        for valid_state in valid_states:
            values = tuple(valid_state[var] for var in variables)
            variable_combinations[variables].add(values)

            if len(variable_combinations[variables]) == 2**len(variables):
                break

    # Now flatten them into tuples
    for key, value in variable_combinations.iteritems():
        int_value = [tuple(map(int, state)) for state in value]
        variable_combinations[key] = tuple(sorted(int_value))
    return variable_combinations

def aggregate_valid_states_frequencies(valid_states, degree=2):
    ''' Given a list of valid substitution dicts, we aggregate all valid states
        according to max_degree

        >>> a, b, c, x, y, z = sympy.symbols('a b c x y z')
        >>> valid_states = [{a: 1, b: 0}, {a: 0, b: 1}, {a: 0, b: 0}]
        >>> for var_comb in aggregate_valid_states_frequencies(valid_states, degree=1).iteritems(): print var_comb
        ((b,), defaultdict(<type 'int'>, {(0,): 2, (1,): 1}))
        ((a,), defaultdict(<type 'int'>, {(0,): 2, (1,): 1}))

        >>> for var_comb in aggregate_valid_states_frequencies(valid_states, degree=2).iteritems(): print var_comb
        ((b, a), defaultdict(<type 'int'>, {(0, 1): 1, (1, 0): 1, (0, 0): 1}))

        >>> valid_states = [{a: 1, b: 0, c: 1}, {a: 0, b: 1, c: 1}, {a: 0, b: 0, c: 1}]
        >>> for var_comb in aggregate_valid_states_frequencies(valid_states, degree=1).iteritems(): print var_comb
        ((b,), defaultdict(<type 'int'>, {(0,): 2, (1,): 1}))
        ((c,), defaultdict(<type 'int'>, {(1,): 3}))
        ((a,), defaultdict(<type 'int'>, {(0,): 2, (1,): 1}))

        >>> for var_comb in aggregate_valid_states_frequencies(valid_states, degree=2).iteritems(): print var_comb
        ((b, a), defaultdict(<type 'int'>, {(0, 1): 1, (1, 0): 1, (0, 0): 1}))
        ((c, a), defaultdict(<type 'int'>, {(1, 0): 2, (1, 1): 1}))
        ((c, b), defaultdict(<type 'int'>, {(1, 0): 2, (1, 1): 1}))

        >>> for var_comb in aggregate_valid_states_frequencies(valid_states, degree=3).iteritems(): print var_comb
        ((c, b, a), defaultdict(<type 'int'>, {(1, 0, 0): 1, (1, 1, 0): 1, (1, 0, 1): 1}))
    '''

    atoms = set.intersection(*map(set, (state.keys() for state in valid_states)))

    variable_combinations = {}

    for variables in itertools.combinations(atoms, r=degree):
        variable_combinations[variables] = defaultdict(int)

    for variables in variable_combinations.iterkeys():
        for valid_state in valid_states:
            values = tuple(valid_state[var] for var in variables)
            variable_combinations[variables][values] += 1

    return variable_combinations

def valid_states_to_simple_binary_deductions(valid_states):
    ''' Take a long list of substitution dicts, all of which are valid states,
        and find any patterns we can

        >>> a, b, c, x, y, z = sympy.symbols('a b c x y z')

        >>> valid_states = [{a: 0, b: 0}, {a: 1, b: 1}, ]
        >>> for var_comb in valid_states_to_simple_binary_deductions(valid_states).iteritems(): print var_comb
        (b, a)

        >>> valid_states = [{a: 1, b: 0}, {a: 0, b: 1}]
        >>> for var_comb in valid_states_to_simple_binary_deductions(valid_states).iteritems(): print var_comb
        (b, -a + 1)

        >>> valid_states = [{a: 0, b: 0}, {a: 0, b: 1}]
        >>> for var_comb in valid_states_to_simple_binary_deductions(valid_states).iteritems(): print var_comb
        (a, 0)

        >>> valid_states = [{a: 1, b: 0}, {a: 0, b: 1}, {a: 0, b: 0}]
        >>> print valid_states_to_simple_binary_deductions(valid_states)
        {}
    '''
    # Intersection will be a set of (variable, value) tuples, where value
    # will be 0 or 1. We can then intersect with other non-contradictory
    # solutions so we are left with deductions that must be true
    intersection = set.intersection(*[set(state.items()) for state in valid_states])

    # Store our deductions somewhere
    deductions = BinaryEquivalenceDict(intersection)

    # difference_grid is a dict of (var1, var2): difference, where var1 and
    # var2 are variables and difference is the deduced difference - None
    # initially when we don't know anything about the relationship.
    # When we get contradictory relations, the tuple will be popped since
    # we can't make any deduction.
    difference_grid = {}

    # Check all the valid states have the same set of variables
    variables = set.intersection(*[set(vs.keys()) for vs in valid_states])

    for vars_ in itertools.combinations(variables, 2):
        difference_grid[vars_] = None

    for state in valid_states:
        # Process the difference relations
        for key, diff in difference_grid.copy().iteritems():
            var1, var2 = key
            # We know they can be equal
            if state[var1] == state[var2]:
                # If they can also be unequal, bin it
                if diff == 1:
                    difference_grid.pop(key)
                else:
                    difference_grid[key] = 0
            else:
                if diff == 0:
                    difference_grid.pop(key)
                else:
                    difference_grid[key] = 1

    # Now update the deductions with differences
    for (var1, var2), diff in difference_grid.iteritems():
        if diff == 1:
            deductions[var1] = sympy.S.One - var2
        elif diff == 0:
            deductions[var1] = var2
        else:
            # This absolutely should not happen as it should be caught
            # by checking intersection
            raise ContradictionException('This should not happen!')

    return deductions

def valid_states_to_deductions(valid_states, max_degree=2):
    ''' Take a long list of substitution dicts, all of which are valid states,
        and find any patterns we can

        >>> a, b, c, x, y, z = sympy.symbols('a b c x y z')

        >>> valid_states = [{a: 0, b: 0}, {a: 1, b: 1}, ]
        >>> for var_comb in valid_states_to_deductions(valid_states, max_degree=2).iteritems(): print var_comb
        (b, a)

        >>> valid_states = [{a: 1, b: 0}, {a: 0, b: 1}]
        >>> for var_comb in valid_states_to_deductions(valid_states, max_degree=2).iteritems(): print var_comb
        (b, -a + 1)

        >>> valid_states = [{a: 0, b: 0}, {a: 0, b: 1}]
        >>> for var_comb in valid_states_to_deductions(valid_states, max_degree=1).iteritems(): print var_comb
        (a, 0)

        >>> valid_states = [{a: 1, b: 0}, {a: 0, b: 1}, {a: 0, b: 0}]
        >>> print valid_states_to_deductions(valid_states, max_degree=1)
        {}
        >>> for var_comb in valid_states_to_deductions(valid_states, max_degree=2).iteritems(): print var_comb
        (a*b, 0)
    '''
    # Keep track of any states we haven't listed above
    unknown_states = set()
    # First munge the states into a {variables: set of valid states} form
    aggregated_states = aggregate_valid_states(valid_states=valid_states, degree=max_degree)

    # Keep a list of the combinations we have searched to save time
    searched = set()

    # Keep a list of all deductions as a {key: set(values)} structure
    deductions = defaultdict(set)

#    N = len(aggregated_states)
#    i = 0
    for variables, value_set in aggregated_states.iteritems():
#        print '{:.1f}'.format(i * 100.0 / N)
#        i += 1
        # Now look for judgements using 1-tuples, 2-tuples, ... max_degree-tuples
        # Until we find a judgement. If we find one, then break
        for deg in xrange(1, max_degree + 1):
            break_ = False
            for indices in itertools.combinations(range(len(variables)), r=deg):
                var = tuple(variables[i] for i in indices)

                # We might end up searching for pairs or singles many times under
                # the current framework, so keep track of it all
                if var in searched:
                    continue
                else:
                    searched.add(var)

                states = tuple(sorted(set([tuple(val[i] for i in indices) for val in value_set])))

                judgement_func = get_deduction_function(states)
                if judgement_func is None:
                    continue
                _deductions = judgement_func(*var)
                if _deductions is not None:
                    for k, v in _deductions:
                        deductions[k].add(v)
                    break_ = True
                    break
            if break_:
                break

    for k, v in deductions.copy().iteritems():
#TODO: Work out how to handle multiple deductions
#        if len(v) > 1:
#            print 'Multiple deductions found'
#            print k, '\t=\t', sorted(v, key=str)
        root_value = v.pop()
        deductions[k] = root_value
#        for _v in v:
#            if not deductions.has_key(_v):
#                deductions[_v] = root_value

    deductions = filter_deductions(deductions)

    return deductions


def valid_states_to_probabalistic_deductions(valid_states, max_degree=2,
                                             filter_simple_binary=False,
                                             filter_definite=False):
    ''' Given a list of valid states, calculate the relative probabilities
        each kind of deduction is true

        >>> a, b, c, x, y, z = sympy.symbols('a b c x y z')

        >>> valid_states = [{a: 0, b: 0}, {a: 0, b: 1}]
        >>> for variables, values in valid_states_to_probabalistic_deductions(valid_states, max_degree=1).iteritems():
        ...     print variables, dict(values)
        b {0: 0.5, 1: 0.5}
        a {0: 1.0}

        >>> valid_states = [{a: 0, b: 0}, {a: 0, b: 1}]
        >>> for variables, values in valid_states_to_probabalistic_deductions(valid_states, max_degree=2).iteritems():
        ...     print variables, dict(values)
        b {0: 0.5, 1: 0.5}
        a {0: 1.0}

        >>> valid_states = [{a: 0, b: 0}, {a: 1, b: 1}]
        >>> for variables, values in valid_states_to_probabalistic_deductions(valid_states, max_degree=2).iteritems():
        ...     print variables, dict(values)
        b {0: 0.5, 1: 0.5, a: 1.0}
        a {0: 0.5, 1: 0.5}

        >>> valid_states = [{a: 1, b: 0}, {a: 0, b: 1}, {a: 0, b: 0}]
        >>> for variables, values in valid_states_to_probabalistic_deductions(valid_states, max_degree=2).iteritems():
        ...     print variables, dict(values)
        a*b {0: 1.0}
        b {0: 0.6666666666666666, 1: 0.3333333333333333, -a + 1: 0.6666666666666666}
        a {0: 0.6666666666666666, 1: 0.3333333333333333}

        Filter anything that isn't a simple binary solution
        >>> valid_states = [{a: 1, b: 0}, {a: 0, b: 1}, {a: 0, b: 0}]
        >>> for variables, values in valid_states_to_probabalistic_deductions(valid_states,
        ...                             max_degree=2, filter_simple_binary=True).iteritems():
        ...     print variables, dict(values)
        b {0: 0.6666666666666666, 1: 0.3333333333333333, -a + 1: 0.6666666666666666}
        a {0: 0.6666666666666666, 1: 0.3333333333333333}


        >>> p16, z3335, z3435, z3637 = sympy.symbols('p16, z3335, z3435, z3637')
        >>> valid_states = [{z3637: 0, p16: 0, z3435: 1, z3335: 0},
        ...                 {z3637: 0, p16: 0, z3435: 0, z3335: 1},
        ...                 {z3637: 0, p16: 1, z3435: 0, z3335: 0}]
        >>> for variables, values in valid_states_to_probabalistic_deductions(valid_states,
        ...                             max_degree=2, filter_simple_binary=False).iteritems():
        ...     print variables, dict(values)
        z3637 {0: 1.0}
        p16 {0: 0.6666666666666666, 1: 0.3333333333333333, -z3335 + 1: 0.6666666666666666}
        p16*z3335 {0: 1.0}
        z3335*z3435 {0: 1.0}
        z3435 {0: 0.6666666666666666, 1: 0.3333333333333333, -z3335 + 1: 0.6666666666666666, -p16 + 1: 0.6666666666666666}
        z3335 {0: 0.6666666666666666, 1: 0.3333333333333333}
        p16*z3435 {0: 1.0}
    '''
    # If we're filtering on simple binary, we can't possibly be interested in
    # cubic interactions
    if filter_simple_binary:
        max_degree = min(max_degree, 2)

    # Store deductions as a {key: {value: frequency}} map
    deductions = defaultdict(lambda: defaultdict(int))

    # First aggregate the states so that we know how frequently each
    # combination occurs
    # We get {key: {value: frequency}} map
    for _degree in xrange(max_degree, 0, -1):

        # Store deductions as a {key: {value: frequency}} map
        deg_deductions = defaultdict(lambda: defaultdict(int))

        frequencies = aggregate_valid_states_frequencies(valid_states,
                                                         degree=_degree)
        # Traverse the tree, converting states into judgements
        for variables, val_freq_dict in frequencies.iteritems():

            super_set = itertools.chain(*(itertools.combinations(val_freq_dict.iteritems(), r=r) for r in xrange(1, len(val_freq_dict) + 1)))
            for subset in super_set:
                values, freq = zip(*subset)
                freq = sum(freq)

                # Pass it as a tuple of states so we can reuse the previous work
                deduc_func = get_deduction_function(values)
                if deduc_func is None:
                    continue

                _deductions = deduc_func(*variables)

                if _deductions is not None:
                    for key, value in _deductions:
                        deg_deductions[key][value] += 1.0 * freq / len(valid_states)

        # Now perform a nested overwrite, so that we throw away incorrect
        # values for lower order deductions when going through the degrees
        # in reverse
        for k1, k2_v in deg_deductions.iteritems():
            if filter_simple_binary and (not is_simple_binary(k1)):
                continue
            for k2, v in k2_v.iteritems():
                if filter_simple_binary and (not is_simple_binary(k2)):
                    continue
                if filter_definite and (v == 1.0):
                    continue
                deductions[k1][k2] = v

    return deductions

def probabalistic_deductions_to_tuples(probabalistic_deductions):
    ''' Munge the results

        >>> p16, z3335, z3435, z3637 = sympy.symbols('p16, z3335, z3435, z3637')
        >>> valid_states = [{z3637: 0, p16: 0, z3435: 1, z3335: 0},
        ...                 {z3637: 0, p16: 0, z3435: 0, z3335: 1},
        ...                 {z3637: 0, p16: 1, z3435: 0, z3335: 0}]
        >>> prob_ded = valid_states_to_probabalistic_deductions(valid_states,
        ...                             max_degree=2, filter_simple_binary=False)
        >>> for t in probabalistic_deductions_to_tuples(prob_ded):
        ...     print t
        (z3637, 0, 1.0)
        (p16, 0, 0.6666666666666666)
        (p16, 1, 0.3333333333333333)
        (p16, -z3335 + 1, 0.6666666666666666)
        (p16*z3335, 0, 1.0)
        (z3335*z3435, 0, 1.0)
        (z3435, 0, 0.6666666666666666)
        (z3435, 1, 0.3333333333333333)
        (z3435, -z3335 + 1, 0.6666666666666666)
        (z3435, -p16 + 1, 0.6666666666666666)
        (z3335, 0, 0.6666666666666666)
        (z3335, 1, 0.3333333333333333)
        (p16*z3435, 0, 1.0)
    '''
    flattened = [[(key, val, prob) for val, prob in values.iteritems()] for key, values in probabalistic_deductions.iteritems()]
    flattened = list(itertools.chain(*flattened))
    return flattened

def filter_deductions(deductions):
    ''' Filter out any deductions that don't reduce degree and introduce new
        variables

        >>> eqns = [
        ...     ('x1*x2', 'x1*x2*x3'),
        ...     ('x + y', 'z - y'),
        ...     ('2*y', 'x + z'),
        ...     ('x', '1 - y')]
        >>> deductions = {sympy.sympify(eqn[0]):
        ...               sympy.sympify(eqn[1]) for eqn in eqns}
        >>> for key, value in deductions.iteritems(): print key, '==', value
        2*y == x + z
        x == -y + 1
        x + y == -y + z
        x1*x2 == x1*x2*x3

        >>> filtered = filter_deductions(deductions)
        >>> for key, value in filtered.iteritems(): print key, '==', value
        x == -y + 1
    '''
    filtered = {}
    for k, v in deductions.iteritems():
        if is_simple_binary(k) and is_simple_binary(v):
            filtered[k] = v
        elif degree(k) < degree(v):
            continue
        elif degree(k) > degree(v):
            filtered[k] = v
        elif v.atoms(sympy.Symbol).issubset(k.atoms(sympy.Symbol)):
            filtered[k] = v
        else:
            continue
    return filtered

if __name__ == '__main__':
    import doctest
    from solver_sequential import SolverSequential
    doctest.testmod()

if 0:#__name__ == '__main__':

    eqns = '''q1 + q2 == z23 + 2*z24
    p3 + 2*q1*q2 + q3 + z23 == 2*z34 + 4*z35
    p3*q1 + p4 + q1*q3 + q2 + q4 + z24 + z34 == 2*z45 + 4*z46
    p3*q2 + p4*q1 + p5 + q1*q4 + q2*q3 + q5 + z35 + z45 == 2*z56 + 4*z57 + 1
    p3*q3 + p4*q2 + p5*q1 + p6 + q1*q5 + q2*q4 + q6 + z46 + z56 == 2*z67 + 4*z68 + 8*z69
    p3*q4 + p4*q3 + p5*q2 + p6*q1 + p7 + q1*q6 + q2*q5 + q7 + z57 + z67 == 8*z710 + 2*z78 + 4*z79
    p3*q5 + p4*q4 + p5*q3 + p6*q2 + p7*q1 + p8 + q1*q7 + q2*q6 + q8 + z68 + z78 == 4*z810 + 8*z811 + 2*z89
    p3*q6 + p4*q5 + p5*q4 + p6*q3 + p7*q2 + p8*q1 + p9 + q1*q8 + q2*q7 + q9 + z69 + z79 + z89 == 2*z910 + 4*z911 + 8*z912 + 1
    p10 + p3*q7 + p4*q6 + p5*q5 + p6*q4 + p7*q3 + p8*q2 + p9*q1 + q1*q9 + q10 + q2*q8 + z710 + z810 + z910 == 2*z1011 + 4*z1012 + 8*z1013 + 1
    p10*q1 + p11 + p3*q8 + p4*q7 + p5*q6 + p6*q5 + p7*q4 + p8*q3 + p9*q2 + q1*q10 + q11 + q2*q9 + z1011 + z811 + z911 == 2*z1112 + 4*z1113 + 8*z1114
    p10*q2 + p11*q1 + p12 + p3*q9 + p4*q8 + p5*q7 + p6*q6 + p7*q5 + p8*q4 + p9*q3 + q1*q11 + q10*q2 + q12 + z1012 + z1112 + z912 == 2*z1213 + 4*z1214 + 8*z1215
    p10*q3 + p11*q2 + p12*q1 + p13 + p3*q10 + p4*q9 + p5*q8 + p6*q7 + p7*q6 + p8*q5 + p9*q4 + q1*q12 + q11*q2 + q13 + z1013 + z1113 + z1213 == 2*z1314 + 4*z1315 + 8*z1316 + 1
    p10*q4 + p11*q3 + p12*q2 + p13*q1 + p14 + p3*q11 + p4*q10 + p5*q9 + p6*q8 + p7*q7 + p8*q6 + p9*q5 + q1*q13 + q12*q2 + q14 + z1114 + z1214 + z1314 == 2*z1415 + 4*z1416 + 8*z1417
    p10*q5 + p11*q4 + p12*q3 + p13*q2 + p14*q1 + p15 + p3*q12 + p4*q11 + p5*q10 + p6*q9 + p7*q8 + p8*q7 + p9*q6 + q1*q14 + q13*q2 + q15 + z1215 + z1315 + z1415 == 2*z1516 + 4*z1517 + 8*z1518 + 1
    p10*q6 + p11*q5 + p12*q4 + p13*q3 + p14*q2 + p15*q1 + p16 + p3*q13 + p4*q12 + p5*q11 + p6*q10 + p7*q9 + p8*q8 + p9*q7 + q1*q15 + q14*q2 + q16 + z1316 + z1416 + z1516 == 2*z1617 + 4*z1618 + 8*z1619 + 16*z1620
    p10*q7 + p11*q6 + p12*q5 + p13*q4 + p14*q3 + p15*q2 + p16*q1 + p17 + p3*q14 + p4*q13 + p5*q12 + p6*q11 + p7*q10 + p8*q9 + p9*q8 + q1*q16 + q15*q2 + q17 + z1417 + z1517 + z1617 == 2*z1718 + 4*z1719 + 8*z1720 + 16*z1721 + 1
    p10*q8 + p11*q7 + p12*q6 + p13*q5 + p14*q4 + p15*q3 + p16*q2 + p17*q1 + p18 + p3*q15 + p4*q14 + p5*q13 + p6*q12 + p7*q11 + p8*q10 + p9*q9 + q1*q17 + q16*q2 + q18 + z1518 + z1618 + z1718 == 2*z1819 + 4*z1820 + 8*z1821 + 16*z1822 + 1
    p10*q9 + p11*q8 + p12*q7 + p13*q6 + p14*q5 + p15*q4 + p16*q3 + p17*q2 + p18*q1 + p3*q16 + p4*q15 + p5*q14 + p6*q13 + p7*q12 + p8*q11 + p9*q10 + q1*q18 + q17*q2 + z1619 + z1719 + z1819 + 1 == 2*z1920 + 4*z1921 + 8*z1922 + 16*z1923
    p10*q10 + p11*q9 + p12*q8 + p13*q7 + p14*q6 + p15*q5 + p16*q4 + p17*q3 + p18*q2 + p3*q17 + p4*q16 + p5*q15 + p6*q14 + p7*q13 + p8*q12 + p9*q11 + 2*q1 + q18*q2 + z1620 + z1720 + z1820 + z1920 == 2*z2021 + 4*z2022 + 8*z2023 + 16*z2024 + 1
    p10*q11 + p11*q10 + p12*q9 + p13*q8 + p14*q7 + p15*q6 + p16*q5 + p17*q4 + p18*q3 + p3*q18 + p4*q17 + p5*q16 + p6*q15 + p7*q14 + p8*q13 + p9*q12 + 2*q2 + z1721 + z1821 + z1921 + z2021 == 2*z2122 + 4*z2123 + 8*z2124 + 16*z2125 + 1
    p10*q12 + p11*q11 + p12*q10 + p13*q9 + p14*q8 + p15*q7 + p16*q6 + p17*q5 + p18*q4 + p3 + p4*q18 + p5*q17 + p6*q16 + p7*q15 + p8*q14 + p9*q13 + q3 + z1822 + z1922 + z2022 + z2122 == 2*z2223 + 4*z2224 + 8*z2225 + 16*z2226 + 1
    p10*q13 + p11*q12 + p12*q11 + p13*q10 + p14*q9 + p15*q8 + p16*q7 + p17*q6 + p18*q5 + p4 + p5*q18 + p6*q17 + p7*q16 + p8*q15 + p9*q14 + q4 + z1923 + z2023 + z2123 + z2223 == 2*z2324 + 4*z2325 + 8*z2326 + 16*z2327 + 1
    p10*q14 + p11*q13 + p12*q12 + p13*q11 + p14*q10 + p15*q9 + p16*q8 + p17*q7 + p18*q6 + p5 + p6*q18 + p7*q17 + p8*q16 + p9*q15 + q5 + z2024 + z2124 + z2224 + z2324 == 2*z2425 + 4*z2426 + 8*z2427 + 16*z2428 + 1
    p10*q15 + p11*q14 + p12*q13 + p13*q12 + p14*q11 + p15*q10 + p16*q9 + p17*q8 + p18*q7 + p6 + p7*q18 + p8*q17 + p9*q16 + q6 + z2125 + z2225 + z2325 + z2425 == 2*z2526 + 4*z2527 + 8*z2528 + 16*z2529 + 1
    p10*q16 + p11*q15 + p12*q14 + p13*q13 + p14*q12 + p15*q11 + p16*q10 + p17*q9 + p18*q8 + p7 + p8*q18 + p9*q17 + q7 + z2226 + z2326 + z2426 + z2526 == 2*z2627 + 4*z2628 + 8*z2629 + 16*z2630 + 1
    p10*q17 + p11*q16 + p12*q15 + p13*q14 + p14*q13 + p15*q12 + p16*q11 + p17*q10 + p18*q9 + p8 + p9*q18 + q8 + z2327 + z2427 + z2527 + z2627 == 2*z2728 + 4*z2729 + 8*z2730 + 16*z2731
    p10*q18 + p11*q17 + p12*q16 + p13*q15 + p14*q14 + p15*q13 + p16*q12 + p17*q11 + p18*q10 + p9 + q9 + z2428 + z2528 + z2628 + z2728 == 2*z2829 + 4*z2830 + 8*z2831
    p10 + p11*q18 + p12*q17 + p13*q16 + p14*q15 + p15*q14 + p16*q13 + p17*q12 + p18*q11 + q10 + z2529 + z2629 + z2729 + z2829 == 2*z2930 + 4*z2931 + 8*z2932 + 1
    p11 + p12*q18 + p13*q17 + p14*q16 + p15*q15 + p16*q14 + p17*q13 + p18*q12 + q11 + z2630 + z2730 + z2830 + z2930 == 2*z3031 + 4*z3032 + 8*z3033
    p12 + p13*q18 + p14*q17 + p15*q16 + p16*q15 + p17*q14 + p18*q13 + q12 + z2731 + z2831 + z2931 + z3031 == 2*z3132 + 4*z3133 + 8*z3134
    p13 + p14*q18 + p15*q17 + p16*q16 + p17*q15 + p18*q14 + q13 + z2932 + z3032 + z3132 == 2*z3233 + 4*z3234
    p14 + p15*q18 + p16*q17 + p17*q16 + p18*q15 + q14 + z3033 + z3133 + z3233 == 2*z3334 + 4*z3335 + 1
    p15 + p16*q18 + p17*q17 + p18*q16 + q15 + z3134 + z3234 + z3334 == 2*z3435 + 4*z3436
    p16 + p17*q18 + p18*q17 + q16 + z3335 + z3435 == 2*z3536 + 1
    p17 + q17 + z3436 + z3536 == 2*z3637
    p18 + q18 + z3637 == 1
    p18 == 1'''.split('\n')
    eqns = eqns[:10] + eqns[-10:]
#    eqns = ['p1 + q1 == 1', 'p1*q2 + q1*p2 == 1 + 2*z2']
    eqns = str_eqns_to_sympy_eqns(eqns)


    from objective_function_helper import equations_to_vanilla_term_dict, reduce_term_dict

    deductions = {}
    for direction in [-1]:#, 1]:
        search = SolverSequential()
        for e in eqns[::direction]:
            search.add_equation(e)
            subbed = search.sub_var(None, max_states=1, verbose=True)
#            print search._length_tuple
            if not subbed:
                break
        valid_states = [s[0].copy() for s in search.valid_states[:]]

#        v = ', '.join(sorted(map(str, valid_states[0].keys())))
#        print '{} = sympy.symbols({})'.format(v, v)
#        for vs in valid_states:
#            print vs

#        _deductions = valid_states_to_deductions(valid_states, max_degree=2)
#        for _ in _deductions.iteritems():
#            print _

        print

        _deductions = valid_states_to_probabalistic_deductions(valid_states, max_degree=2, filter_simple_binary=False)
        for _ in _deductions.iteritems():
            print _

#        deductions.update(_deductions)
#        for d in _deductions.iteritems(): print d

#    raise Exception
#    term_dict = equations_to_vanilla_term_dict(eqns)
#    new_term_dict = reduce_term_dict(term_dict.copy(), deductions, penalty_factor=0)
#    print count_qubit_interactions(term_dict)
#    print count_qubit_interactions(new_term_dict)






#num1, num2, prod = 4, 4, 143
#eqns = filter(is_equation, generate_carry_equations(num1, num2, prod))
#search = SolverSequential(eqns)
#while search.sub_var(4):
#    pass
#
#valid_states = [s[0].copy() for s in search.valid_states[:]]
#
#t = valid_states_to_deductions(valid_states)
#print t
