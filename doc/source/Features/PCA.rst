.. _PCA:
.. currentmodule:: LaserPy

PCA
===

Principle Component Analysis on regions of 3D points.  This is sort of like
fitting an ellipsoid to a region.  It provides good surface normals, and can be
used to distinguish flat, thin and scattered regions (like tree canopies), as
in [lalonde2006natural]_. It's likely the fastest feature here, the main slow-down is in
selecting the 3D regions in the first place.

Also note that the points in your 3D region should be representative of the
shape- in velodyne data, some regions on the ground are just a single line,
which PCA will interpret as a thin linear region. See [quadros2012feature]_ for
a simple check, implemented in :func:`minRadiusSelection`.

PCA can be used to generate :mod:`surface normals
<pyception.algorithms.surf_norms>`, as well as :ref:`keypoints <pca-keypoints>`
at highly flat or linear regions, as in [quadros2012feature]_.

.. [lalonde2006natural] J.F. Lalonde, N. Vandapel, D.F. Huber, and M. Hebert.
   Natural terrain classiﬁcation using three-dimensional ladar data for ground
   robot mobility. Journal of Field Robotics, 23(10):839–861, 2006.

.. autoclass:: PCA
   :members:


minRadiusSelection
^^^^^^^^^^^^^^^^^^

.. autofunction:: minRadiusSelection


