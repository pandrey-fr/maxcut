# coding: utf-8

"""Utility functions to instantiate example graphs."""

import networkx as nx
import numpy as np


def load_gset_graph(path):
    """Create a networkx.graph based on a Gset graph file.

    Gset is a collection of graphs used in by Burer and Monteiro
    in papers from 2003 and 2005, available for download at
    https://web.stanford.edu/~yyye/yyye/Gset.
    """
    graph = nx.Graph()
    with open(path) as file:
        n_nodes = int(file.readline().split(' ', 1)[0])
        graph.add_nodes_from(range(n_nodes))
        for row in file:
            start, end, weight = [int(e) for e in row.strip('\n').split()]
            graph.add_edge(start - 1, end - 1, weight=weight)
    return graph


def generate_sbm(sizes, probs, maxweight=1):
    """Generate a Stochastic Block Model graph.

    Assign random values drawn from U({1, ...,  maxw}) to the edges.

    sizes     : list of sizes (int) of the blocks
    probs     : matrix of probabilities (in [0, 1]) of edge creation
                between nodes depending on the blocks they belong to
    maxweight : maximum value of the weights to randomly assign
                (default 1, resulting in weights all equal to 1)
    """
    graph = nx.stochastic_block_model(sizes, probs)
    weights = 1 + np.random.choice(maxweight, len(graph.edges))
    weights = dict(zip(graph.edges, weights))
    nx.set_edge_attributes(graph, weights, 'weight')
    return graph
