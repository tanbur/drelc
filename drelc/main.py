
import networkx
import os
import sys

from local_search import local_search
from term_dict import TermDict
from variable_graph import VariableGraph, find_subproblems
from judgement_sniffer import valid_states_to_deductions
from objective_function_helper import reduce_term_dict_with_deductions, term_dict_to_coef_string

def main(filepath):
    ''' Reduce an objective function using deduc-reduc and ELCs '''
    coef_str = open(filepath, 'r').read()

    # Put a limit on the number of coefficients we read
    # coef_str = '\n'.join(coef_str.split('\n')[:100])

    # Compile the term dict
    term_dict = TermDict.from_coef_str(coef_str)
    reduced = reduce_term_dict(term_dict)

    # Use the helper function, as TermDicts can't assign their own variables IDs yet
    print term_dict_to_coef_string(reduced)

def reduce_term_dict(term_dict):
    ''' Given a term dict, return a reduced one '''
    # Create a graph from the variables
    graph = VariableGraph.from_term_dict(term_dict)

    # Make sure we only have one connected component
    if not networkx.algorithms.is_connected(graph):
        raise ValueError('We only want connected graphs at the moment')

    # Now extract subproblems that we want to examine. These are our local-search starting points
    subproblems = find_subproblems(graph, max_interior_nodes=10, max_boundary_nodes=10)

    if not len(subproblems):
        print 'No subproblems found!'
        return
    else:
        print 'Starting {} local searches'.format(len(subproblems))


    all_deductions = []
    for problem in subproblems:
        minima = local_search(problem, term_dict.variables_to_terms(problem))
        deductions = valid_states_to_deductions(minima)
        print '{} deductions found'.format(len(deductions))
        all_deductions.extend(deductions)

    reduced = reduce_term_dict_with_deductions(term_dict=term_dict, deductions=dict(all_deductions), lagrangian_coefficient=0,
                                     preserve_terms=False, substitute_deductions=True)
    return reduced


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.path.join('coef_strs', '20x20_016_0_coef.txt')
    main(filepath)