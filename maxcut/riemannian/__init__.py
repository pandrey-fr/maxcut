# coding: utf-8

"""Collection of classes and functions to operate on Riemann manifolds.

The functionalities implemented here are meant to serve as backend
resources for the Bureir-Monteiro approach of the Max-Cut problem
(see maxcut.MaxCutBM for details).
"""

from . import stiefel
from ._tcg import TruncatedConjugateGradient
from ._rtr import RiemannianTrustRegion
