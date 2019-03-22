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

from ._stiefel import (
    inner_prod, froebenius, random_from_stiefel, stiefel_dimension,
    stiefel_projection, stiefel_retraction, symblockdiag
)
