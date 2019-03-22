# coding: utf-8

"""Riemannian trust-region algorithm to solve SDP problems."""

import numpy as np

from maxcut.riemannian import TruncatedConjugateGradient
from maxcut.riemannian.stiefel import (
    inner_prod, froebenius, random_from_stiefel, stiefel_dimension,
    stiefel_projection, stiefel_retraction, symblockdiag
)


class RiemannianTrustRegion:
    """Riemannian trust-region algorithm to solve SDP problems.

    The Riemannian trust-regions method implemented here is meant
    to find a minimizer of $$<CY, Y>$$ where $C$ is a given matrix
    of dimensions $(n, n)$ and $Y \\in St(1, n)^p$, i.e. $Y$
    is a matrix of dim $(n, p)$ so that $diag(Y Y^\\top) = 1_{{R^n}}$.
    In this context, $<X, Y> = Tr(X^\\top Y)$.

    This problem is a non-convex relaxation of the Semi-Definite
    Programming problem of minimizing $<C, X>$ where $X$ is PSD
    and all elements of its diagonal are 1, by rewriting $X$ as
    $Y Y^\\top$ with the only constraint that $Y \\in St(1, n)^p$.

    Above, we denote by $St(d, p)^n$ an extension of the usual
    Stiefel manifold to matrices made of p blocks which belong
    to a $St(d, p)$ manifold. Here, since d=1, it simply means
    that the p columns of the matrices belong to a $St(1, p)$
    manifold, hence yielding the aforementionned constraint on
    its diagonal. (see [Boumal, 2016])

    The algorithm and the notations are based on [Algorithm 10]
    presented in [Absil et al. 2008] (see full reference below),
    while default parameters and implementation are partly based
    on the RTR Matlab implementation from the Manopt toolbox
    (see [Boumal et al. 2014] and link to source code below).

    Usage:
    >>> rtr = RiemannianTrustRegion(cost_mat, dim_p)
    >>> minimizer = rtr.get_solution()

    References:
    P.-A. Absil, R. Mahony, and R. Sepulchre (2008). Optimization
    Algorithms on Matrix Manifolds. Princeton University Press.

    N. Boumal (2016). A Riemannian low-rank method for optimization
    oversemidefinite matrices with block-diagonal constraints.
    arXiv preprint.

    N. Boumal, B. Mishra, P.-A. Absil and R. Sepulchre (2014).
    Manopt, a Matlab toolbox for optimization on manifolds.
    Journal of Machine Learning Research.
    source code at https://github.com/NicolasBoumal/manopt
    """

    def __init__(
            self, cost_mat, dim_p, rho_prime=.1, maxiter=1000, **kwargs
        ):
        """Instantiate the RTR solver, setting the problem.

        The problem to solve is $min <CY, Y>$ for a given square
        matrix $C$ of dimension n*n, with $Y \\in St(1, p)^n$
        i.e. $Y$ a matrix of dimensions n*p so that $Y Y^\\top$
        has only values equal to 1 on its diagonal.

        problem-defining arguments
        --------------------------
        cost_mat  : square matrix defining the cost function
        dim_p     : dimension p of the St(1, n)^p manifold
                    on which to look for a minimizer

        method-adjusting arguments
        --------------------------
        rho_prime : minimum value of ratio rho at step k
                    to update the value of the candidate
        maxiter   : maximum number of RTR iterations
        **kwargs  : keyword arguments to pass to the trust-region
                    subproblem solvers, in {{'theta', 'kappa'}}.
                    (see TruncatedConjugateGradient for details)

        See [Absil et al. 2008] for details on the algorithm.
        Some parameters, such as the mean and starting radius
        of the trust-regions in which to look for an optimum,
        are set according to the default values used in the
        Matlab implementation of the algorithm in the Manopt
        toolbox (see [Boumal et al. 2014]).
        """
        self.cost_mat = cost_mat
        self.dimensions = (len(cost_mat), dim_p)
        # Set up method parameters.
        self.rho_prime = min(rho_prime, .25)
        self.maxiter = maxiter
        # Set up deltabar and the max number of tCG iterations
        # based on the dimension of the Stiefel manifold.
        self.deltabar = np.sqrt(np.prod(self.dimensions))
        kwargs['maxiter'] = stiefel_dimension(*self.dimensions)
        self.tcg_kwargs = kwargs
        # Set up a private attribute to store the solutions.
        self.__candidates = []

    def get_candidates(self, verbose=False):
        """Return the lazy-evaluated candidates reached by the RTR method."""
        if not self.__candidates:
            self.solve(verbose)
        return self.__candidates

    def solve(self, verbose=True):
        """Run the RTR algorithm to find candidate minimizers."""
        # Set up the initial trust-regions radius (follow manopt default).
        delta = self.deltabar / 8  # default value in manopt implementation
        # Pick a random value x_0 and compute f values on it.
        x_k = random_from_stiefel(*self.dimensions)
        value, gradient, get_hessian = self._get_f_values(x_k)
        # Iteratively solve and adjust trust-region subproblems.
        n_iter = 0
        stop_cause = 'maximum number of iterations reached'
        for n_iter in range(self.maxiter):
            # Solve the trust-region subproblem.
            eta_k, rho_k, x_new = self._solve_tr_subproblem(
                x_k, value, gradient, get_hessian, delta
            )
            # Update the radius of the trust region (delta), if relevant.
            if rho_k < .25:
                delta *= .25
            elif rho_k > .75 and round(froebenius(eta_k), 2) == delta:
                delta = min(2 * delta, self.deltabar)
            # Adopt the new candidate solution, if relevant.
            if rho_k > self.rho_prime:
                value_new, gradient, get_hessian = self._get_f_values(x_new)
                x_k, value = x_new, value_new
                self.__candidates.append(x_k)
                # Check a stopping criterion on the gradient's norm.
                if froebenius(gradient) < 1e-6:
                    stop_cause = "vanishing gradient"
                    break
        # Optionally print-out stopping cause.
        if verbose:
            print(stop_cause)
            print(
                "%i candidate solutions reached after %i iterations.\n"
                % (len(self.__candidates), n_iter)
            )

    def _get_f_values(self, matrix):
        """Return f(matrix), grad f(matrix) and Hess f(matrix)[.].

        The values of f and of its riemannian gradient are explicitely
        computed, while a function is built to compute the riemannian
        hessian in any given direction.
        """
        # Compute the value of f.
        matprod = np.dot(matrix, matrix.T)
        value = inner_prod(self.cost_mat, matprod)
        # Compute the value of grad f.
        gradient = np.dot(self.cost_mat, matrix)
        gradient = 2 * stiefel_projection(matrix, gradient)
        # Define a function to compute the value of Hess f in a direction.
        sym = symblockdiag(np.dot(self.cost_mat, matprod))
        def get_hessian(direction):
            """Get the riemannian hessian of f in a given direction."""
            nonlocal self, sym, matrix
            hess = np.dot(self.cost_mat, direction) - np.dot(sym, direction)
            return 2 * stiefel_projection(matrix, hess)
        # Return f(x_k), grad f(x_k) and function to get Hess f(x_k)[d].
        return value, gradient, get_hessian

    def _solve_tr_subproblem(self, x_k, value, gradient, get_hessian, delta):
        """Solve the trust-region subproblem and return derived quantities.

        x_k         : current candidate optimum x_k
        value       : value of f(x_k)
        gradient    : value of grad f(x_k)
        get_hessian : function to get Hess f(x_k) [.]
        delta       : radius of the trust-region

        The trust-region subproblem consists in minimizing the quantity
        $m(\\eta) = f(x_k) + <\\text{{grad}} f(x_k), eta>
                    + .5 <\\text{{Hess}} f(x_k)[eta], eta>$
        under the constraint $||\\eta|| = \\Delta$.

        First, estimate eta_k using the truncated conjugate gradient
        method, and gather the value of the cost function in eta_k.

        Then, compute the retraction of eta_k from the tangent space
        in x_k; this retraction is a candidate for $x_{{k+1}}$.

        Finally, compute the ratio rho_k used to accept of reject
        the previous candidate and update the radius delta:
        $\\rho_k = (f(x_k) - f(R_x(\\eta_k))) / (m(0) - m(\\eta_k))$.

        Return eta_k, rho_k and R_x(eta_k) (candidate eta_{{k+1}}).
        """
        # Arguments serve readability; pylint: disable=too-many-arguments
        # Solve the trust-region subproblem.
        tcg = TruncatedConjugateGradient(
            value, gradient, get_hessian, delta, **self.tcg_kwargs
        )
        eta_k, cost_eta = tcg.get_solution()
        # Compute the retraction of the solution.
        rx_k = stiefel_retraction(x_k, eta_k)
        # Compute the ratio rho_k.
        f_rxk = inner_prod(self.cost_mat, np.dot(rx_k, rx_k.T))
        rho_k = (value - f_rxk) / (value - cost_eta + 1e-30)
        # Return eta_k, rho_k and the potential new optimum candidate.
        return eta_k, rho_k, rx_k
