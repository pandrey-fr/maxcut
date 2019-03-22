# coding: utf-8

"""Truncated Conjugate Gradient method to solve the Trust-Region subproblem."""

import numpy as np

from maxcut.riemannian.stiefel import inner_prod, froebenius


class TruncatedConjugateGradient:
    """Truncated Conjugate Gradient for the Trust-Region subproblem.

    The truncated Conjugate-Gradient method implemented here
    is meant to approximate the solution of the trust-region
    subproblem, i.e. minimizing the following quadratic form
    $$f(x_k) + <\\text{{grad}}f(x_k), \\eta> + .5 <H_k[\\eta, \\eta]>$$
    in $\\eta$ subject to $||eta|| \\leq \\Delta$, where $f$, its
    riemannian gradient and riemannian hessian (used as $$H$$)
    are pre-computed, $\\Delta$ is given and the scalar product
    and norm are implemented (outside from this class) as (resp.)
    $<X, Y> = Tr(X^\\top Y)$ and the Froebenius norm.

    The algorithm and the notations are based on [Algorithm 11]
    presented in [Absil et al. 2008] (see full reference below),
    while default parameters and implementation are partly based
    on the tCG Matlab implementation from the Manopt toolbox
    (see [Boumal et al. 2014] and link to source code below).

    The truncation of the gradient descent is designed using
    the following stopping criterion :
    $$||r_{{j+1}}|| \\leq ||r_0|| \\min(||r_0||^\\theta, \\kappa)$$
    where $r_j$ is a residual at step $j$ with $r_0$ equal to
    $\\text{{grad}}f(x_k)$, and $\\theta and \\kappa$ are given
    parameters. This criterion guarantees superlinear convergence
    (see equation [7.10] from [Absil et al. 2008]).

    Usage:
    >>> tcg = TruncatedConjugateGradient(value, gradient, get_hess, radius)
    >>> minimizer, min_cost = tcg.get_solution()

    References:
    P.-A. Absil, R. Mahony, and R. Sepulchre (2008). Optimization
    Algorithms on Matrix Manifolds. Princeton University Press.

    N. Boumal, B. Mishra, P.-A. Absil and R. Sepulchre (2014).
    Manopt, a Matlab toolbox for optimization on manifolds.
    Journal of Machine Learning Research.
    source code at https://github.com/NicolasBoumal/manopt
    """

    # Attrib. serve readability; pylint: disable=too-many-instance-attributes
    def __init__(
            self, value, gradient, get_hess, radius,
            theta=1, kappa=.1, maxiter=1000
        ):
        """Instantiate the tCG solver, setting the problem's quantities.

        problem-defining arguments
        --------------------------
        value    : value of f in x_k
        gradient : riemannian gradient of f in x_k
        get_hess : function to compute the riemannian hessian
                   of f(x_k) in a given direction
        radius   : constraint value of the norm of the solution

        method-adjusting arguments
        --------------------------
        theta    : value of theta in the stopping criterion
        kappa    : value of kappa in the stopping criterion
        maxiter  : maximum number of iterations
        """
        # Arguments serve modularity; pylint: disable=too-many-arguments
        # Assign the problem's quantities as attributes.
        self.value = value
        self.gradient = gradient
        self.get_hessian = get_hess
        self.radius = radius
        # Set up the stopping criterion.
        bound = froebenius(gradient)
        self.stopping_criterion = (bound * min(bound ** theta, kappa)) ** 2
        self.maxiter = maxiter
        # Set up private solution attributes.
        self.__solution = None
        self.__cost = np.inf

    def get_solution(self, verbose=False):
        """Return the lazy-evaluated solution reached by the tCG method.

        Return both the estimated minimizer of the cost function
        and the value of the latter in that point.
        """
        if self.__solution is None:
            self.solve(verbose)
        return self.__solution, self.__cost

    def solve(self, verbose=True):
        """Solve the trust region subproblem using the tCG method."""
        # Refactoring is hard here; pylint: disable=too-many-locals
        # Initialize quantities used in solving the TR subproblem.
        eta, delta, resid, r_norm = self._initialize_solver()
        self.__cost = self.value
        stop_cause = 'maximum number of iterations reached'
        # Iteratively update eta until convergence (or stop).
        n_iter = 0
        for n_iter in range(self.maxiter):
            # Compute <delta, Hess f[delta]> and thus update factor alpha.
            hess_delta = self.get_hessian(delta)
            dh_norm = inner_prod(delta, hess_delta)
            alpha = r_norm / dh_norm
            # Compute candidate eta_new and approximate Hess f [eta_new].
            eta_new = eta + alpha * delta
            # If <delta, Hess f[delta]>  <= 0 or ||eta_{k+1}|| >= Delta,
            # compute an update of eta of proper norm and stop iterating.
            if (dh_norm <= 0) or (froebenius(eta_new) >= self.radius):
                eta = self._fit_eta_tau(eta, delta)
                stop_cause = 'radius condition -- computed fitted-norm eta'
                break
            # Check that the cost decreases with eta_new, otherwise stop.
            new_cost = self._get_model_cost(eta_new)
            if new_cost >= self.__cost:
                stop_cause = 'increased cost -- adopted previous solution'
                break
            self.__cost = new_cost
            # Update the residuals and check the stopping criterion.
            resid += alpha * hess_delta
            rnorm_new = inner_prod(resid, resid)
            if rnorm_new <= self.stopping_criterion:
                stop_cause = 'stopping criterion reached'
                break
            # In the absence of solution, update quantities and iterate.
            delta = - resid + delta * rnorm_new / r_norm
            eta, r_norm = eta_new, rnorm_new
        # Store the solution reached as well as the cost function on it.
        self.__solution = eta
        self.__cost = self._get_model_cost(eta)
        # Print-out the number of iterations needed to converge.
        if verbose:
            print("Solution reached afer %i iterations." % n_iter)
            print(stop_cause)

    def _initialize_solver(self):
        """Set up and return the initial quantities used by the solver."""
        eta = np.zeros_like(self.gradient)
        delta = - self.gradient
        resid = self.gradient
        r_norm = inner_prod(resid, resid)
        return eta, delta, resid, r_norm

    def _get_model_cost(self, eta):
        """Return the cost function's value, given eta and Hessf[eta]."""
        return (
            self.value + inner_prod(self.gradient, eta)
            + .5 * inner_prod(self.get_hessian(eta), eta)
        )

    def _fit_eta_tau(self, eta, delta):
        """Compute the problem's solution in a specific case.

        Given eta and delta at a given iteration, compute tau
        so that $||\\eta_j + \\tau \\delta_j|| = \\Delta$ and
        return the resulting eta of norm Delta (self.radius).

        This problem can be rewritten as an order-2 polynom
        in tau whose discriminant is strictly positive.
        A positive root is thus computed explicitely.
        """
        coeff_a = np.sum(np.square(delta))
        coeff_b = 2 * np.sum(eta * delta)
        coeff_c = np.sum(np.square(eta)) - (self.radius ** 2)
        discrim = (coeff_b ** 2) - (4 * coeff_a * coeff_c)
        tau = .5 * (np.sqrt(discrim) - coeff_b) / coeff_a
        return eta + tau * delta
