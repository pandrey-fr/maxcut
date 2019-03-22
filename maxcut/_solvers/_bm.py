# coding: utf-8

"""Bureir-Monteiro approach solver for the Max-Cut problem."""

import networkx as nx
import numpy as np

from maxcut.riemannian import RiemannianTrustRegion
from maxcut._solvers.backend import (
    AbstractMaxCut, get_partition, get_cut_value
)


class MaxCutBM(AbstractMaxCut):
    """Bureir-Monteiro approach solver for the Max-Cut problem.

    Given a graph with non-negative weights, the method implemented
    here aims at minimizing $$<CY, Y>$$, where C denotes the adjacency
    matrix of the graph and $Y$ is a matrix of dimensions (n, p) so
    that each of its rows is of unit norm.

    The implementation relies on a Riemannian Trust-Region algorithm,
    which itself relies on a truncated conjugate gradient method to
    iteratively solve and adjust trust-region subproblems in order
    to converge towards a global minimizer.

    Usage:
    >>> bm = MaxCutBM(graph)
    >>> cut = bm.get_solution('cut')          # solve problem here
    >>> cut_value = bm.get_solution('value')  # get pre-computed solution

    See the documentation of classes RiemannianTrustRegion and
    TruncatedConjugateGradient for more details and references.
    """

    def __init__(self, graph, dim_p=None, **kwargs):
        """Instantiate the Bureir-Monteiro Max-Cut solver.

        graph    : networkx.Graph instance of the graph to cut
        dim_p    : optional value of p; otherwise, use
                   ceil(sqrt(2 * n_nodes))
        **kwargs : any keyword argument for RiemannianTrustRegion
                   may additionally be passed (e.g. maxiter)
        """
        # Declare the graph attribute and the __results backend one.
        super().__init__(graph)
        # Set up the dimension of the search space.
        if dim_p is None:
            dim_p = np.ceil(np.sqrt(2 * len(graph)))
        self.dim_p = int(dim_p)
        # Store arguments to pass on to the RiemannianTrustRegion solver.
        self._kwargs = kwargs

    def solve(self, verbose=True):
        """Solve the BM-formulated max-cut problem using RTR.

        Resulting cut, value of the cut and solved matrix
        may be accessed through the `get_solution` method.
        """
        # Run the RTR algorithm and gather candidate solutions.
        adjacent = nx.adjacency_matrix(self.graph).toarray()
        rtr = RiemannianTrustRegion(adjacent, self.dim_p, **self._kwargs)
        candidates = rtr.get_candidates(verbose)
        # Find and keep the best candidate.
        matrix, cut, value = self._get_best_candidate(candidates)
        self._results = {'matrix': matrix, 'cut': cut, 'value': value}
        # Optionally be verbose about the results.
        if verbose:
            print(
                "Solved the BM-formulated max-cut problem.\n"
                "Solution cuts off %f share of total weights." % value
            )

    def _get_best_candidate(self, candidates):
        """Select the best solution among a series of candidates.

        Return both the matrix, derived partition and value of
        the latter associated with the best candidate (in terms
        of sum of weights of the edges cut off).
        """
        # Get the partition defined by each candidate and their cut-value.
        partitions = [get_partition(vectors) for vectors in candidates]
        scores = [get_cut_value(self.graph, cut) for cut in partitions]
        # Select the best candidate and return it.
        best = np.argmax([scores])
        return candidates[best], partitions[best], scores[best]
