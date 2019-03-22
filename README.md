# maxcut: Max-Cut problem solving tools following a variety of approaches

Implementation realized in the context of scholar project on the max-cut
problem.

#### Implemented Max-Cut solving approaches

`maxcut.MaxCutSDP` interfaces external solvers (such as SCS or CVXOPT)
to solve the SDP (Semi-Definite Programming) formulation of the Max-Cut
optimization problem.

`maxcut.MaxCutBM` implements the Burer-Monteiro approach, which consists
in using a riemannian trust-regions algorithm to solve a non-convex
formulation of the Max-Cut problem.

#### References

N. Boumal, V. Voroninski and A. Bandeira (2016). The non-convex Burer-Monteiro
approach works on smooth semidefinite programs. In proceedings NIPS 2016.
[available here](https://arxiv.org/abs/1606.04970)

N. Boumal (2016). A Riemannian low-rank method for optimization
oversemidefinite matrices with block-diagonal constraints.
arXiv preprint. [available here](https://arxiv.org/abs/1506.00575)

N. Boumal, B. Mishra, P.-A. Absil and R. Sepulchre (2014).
Manopt, a Matlab toolbox for optimization on manifolds.
_Journal of Machine Learning Research_.
[matlab source code](https://github.com/NicolasBoumal/manopt)

P.-A. Absil, R. Mahony, and R. Sepulchre (2008). Optimization
Algorithms on Matrix Manifolds. Princeton University Press.

#### License

Copyright 2019 Paul Andrey

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
