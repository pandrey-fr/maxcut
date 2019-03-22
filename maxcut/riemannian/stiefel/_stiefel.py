# coding: utf-8

"""Collection of operations relative to St(1, p)^n Stiefel manifolds.

A Stiefel manifold St(n, p) is a riemannian manifold defined
as the ensemble of matrices of dimensions (n, p) so that all
$X$ in St(n, p) verify $X^\\top X = I_p$.

Here, the operations concern a more specific and less constraining
case, based on [Boumal, 2016]. We denote by St(1, p)^n the ensemble
of matrices whose columns belong to a St(1, p) manifold; hence we
have $diag(Y Y^\\top) = 1_{{R^n}}$ for any $Y \\in St(1, p)^n.$

Reference:
N. Boumal (2016). A Riemannian low-rank method for optimization
oversemidefinite matrices with block-diagonal constraints.
arXiv preprint.
"""

import numpy as np


def inner_prod(mat_x, mat_y):
    """Compute the inner product between two matrices.

    $<X, Y> := Tr(X^\\top Y)$ with $X$ and $Y$ two matrices with
    the same dimensions. This inner product is notably that used
    on Stiefel manifolds.
    """
    return np.trace(np.dot(mat_x.T, mat_y))


def froebenius(matrix):
    """Compute the Froebenius norm of a matrix.

    $$||X||_F := \\sqrt{{Tr(X^\\top X)}}$$
    is notably the norm used on Stiefel manifolds.
    """
    return np.sqrt(np.trace(np.dot(matrix.T, matrix)))


def random_from_stiefel(dim_n, dim_p):
    """Generate a random matrix from the St(1, p)^n manifold."""
    matrix = np.random.normal(size=(dim_n, dim_p))
    norm = np.sqrt(np.sum(np.square(matrix), axis=1))
    return matrix / np.expand_dims(norm, 1)


def stiefel_dimension(dim_n, dim_p):
    """Return the dimension of a Stiefel manifold St(1, p)^n.

    in general, dim St(d, p)^m = mdp - .5 * md(d + 1)
    hence here, dim St(1, p)^n = np - n = n(p - 1)
    """
    return dim_n * (dim_p - 1)


def stiefel_projection(pointmat, matrix):
    """Project orthogonally a matrix on the tangent space at a point.

    This projection is defined for matrices in a St(1, p)^n manifold.

    pointmat : matrix at which the tangent space is defined
    matrix   : matrix to project on the tangent space
    """
    sym = symblockdiag(np.dot(matrix, pointmat.T))
    return matrix - np.dot(sym, pointmat)


def stiefel_retraction(pointmat, matrix):
    """Retract a matrix from a tangent space to the manifold.

    This retraction is defined for matrices in a St(1, p)^n manifold.

    pointmat : matrix at which the tangent space is defined
    matrix   : matrix to retract from the tangent space
    """
    def retract_row(row):
        svd = np.linalg.svd(np.expand_dims(row, 0), full_matrices=False)
        return np.dot(svd[0], svd[2])
    return np.concatenate([retract_row(row) for row in pointmat + matrix])


def symblockdiag(matrix):
    """Extract the diagonal of a matrix, setting over values to zero.

    This is a specific case of the symblockdiag transformation
    defined in [Boumal, 2016] for matrices that belong to a St(1, p)^n
    manifold (instead of a more general St(d, p)^m one).
    """
    return np.diag(np.diag(matrix))
