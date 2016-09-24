import itertools

import networkx


class VariableGraph(networkx.MultiGraph):
    ''' Build up a graph '''

    @classmethod
    def from_term_dict(cls, term_dict):
        ''' Return a graph using term_dict adjacencies '''
        graph = cls()

        for _var, _coef in term_dict.iteritems():
            edges = [(var1, var2, _coef) for var1, var2 in itertools.combinations(_var, 2)]
            graph.add_weighted_edges_from(edges)
        networkx.freeze(graph)  # Prevent future changes
        return graph

    def draw(self):
        ''' Pop up a quick graphic '''
        from matplotlib import pyplot
        networkx.draw(self, with_labels=True)
        pyplot.show()

def find_subproblems(graph, max_interior_nodes, max_boundary_nodes):
    ''' Find a list of subgraphs, each of which has size less than max_interior_nodes and a node boundary
        no bigger than max_boundary_nodes.
    '''
    subproblems = []
    nodes_examined = set()
    for node in graph.nodes_iter():
        # If we have already got this node in a subproblem, move on
        if node in nodes_examined:
            continue

        subproblem = expand_subproblem(graph, set([node]), max_interior_nodes=max_interior_nodes, max_boundary_nodes=max_boundary_nodes)

        # If the subproblem comes back as None, then continue
        if subproblem is None:
            continue

        # Now we have a new subproblem, add it on to the list
        subproblems.append(subproblem)
        nodes_examined.update(set(subproblem))
    return subproblems

def expand_subproblem(graph, nodes, max_interior_nodes, max_boundary_nodes):
    ''' Given a subproblem, try and expand it within the constraints given by max_max_interior_nodes and
        max_max_boundary_nodes. Currently expansion works on a greedy basis.
        Function should return a single subproblem, or None if no such problem exists.
    '''
    # If we're already at the maximum, then we will find nothing here
    boundary_nodes = networkx.algorithms.boundary.node_boundary(graph, nodes)
    if (len(nodes) > max_interior_nodes) or (len(boundary_nodes) > max_boundary_nodes):
        return None

    # For each of the nodes in the node boundary, see how big the node boundary would be if we added it on
    candidate_subproblems = []
    for node in boundary_nodes:
        candidate_nodes = nodes.copy()
        candidate_nodes.add(node)
        candidate_nodes = expand_subproblem(graph=graph, nodes=candidate_nodes, max_interior_nodes=max_interior_nodes,
                                            max_boundary_nodes=max_boundary_nodes)
        # We found no joy
        if candidate_nodes is None:
            continue
        else:
            candidate_subproblems.append(candidate_nodes)

    if not len(candidate_subproblems):
        return nodes
    else:
        return max(candidate_subproblems, key=len)

