.. _ICP:
.. currentmodule:: LaserPy


ICP
===

Iterative closest points algorithms.

Point-to-plane
--------------

These solve point-to-plance ICP by linearizing the rotations and solving a least
squares linear equation.  Presumes target and template points are already
mean-centred.  Based on
http://www.cs.princeton.edu/~smr/papers/icpstability.pdf

.. autofunction:: ICP_PointPlane_2D

.. autofunction:: ICP_PointPlane_3D

